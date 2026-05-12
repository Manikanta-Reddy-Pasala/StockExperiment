# Exide Industries Ltd. (EXIDEIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 361.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 161 |
| ALERT2 | 158 |
| ALERT2_SKIP | 72 |
| ALERT3 | 475 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 182 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 183 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 200 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 144
- **Target hits / Stop hits / Partials:** 5 / 183 / 12
- **Avg / median % per leg:** 0.05% / -0.56%
- **Sum % (uncompounded):** 10.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 108 | 26 | 24.1% | 5 | 103 | 0 | -0.30% | -32.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.96% | -3.9% |
| BUY @ 3rd Alert (retest2) | 104 | 26 | 25.0% | 5 | 99 | 0 | -0.27% | -28.1% |
| SELL (all) | 92 | 30 | 32.6% | 0 | 80 | 12 | 0.46% | 42.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.59% | -1.2% |
| SELL @ 3rd Alert (retest2) | 90 | 30 | 33.3% | 0 | 78 | 12 | 0.49% | 44.0% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.84% | -5.0% |
| retest2 (combined) | 194 | 56 | 28.9% | 5 | 177 | 12 | 0.08% | 15.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 12:15:00 | 188.10 | 189.61 | 189.73 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 14:15:00 | 190.65 | 189.88 | 189.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 193.30 | 190.56 | 190.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 15:15:00 | 205.50 | 205.74 | 202.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-19 09:15:00 | 203.00 | 205.74 | 202.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 201.65 | 204.92 | 202.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:00:00 | 201.65 | 204.92 | 202.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 202.85 | 204.51 | 202.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:30:00 | 201.20 | 204.51 | 202.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 202.30 | 204.07 | 202.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 11:30:00 | 202.95 | 204.07 | 202.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 12:15:00 | 202.95 | 203.84 | 202.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 13:45:00 | 203.60 | 203.78 | 202.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 12:15:00 | 201.55 | 202.79 | 202.56 | SL hit (close<static) qty=1.00 sl=201.80 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 14:15:00 | 201.90 | 202.37 | 202.40 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 206.50 | 203.14 | 202.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 207.20 | 203.95 | 203.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 206.65 | 207.45 | 205.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 12:00:00 | 206.65 | 207.45 | 205.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 207.85 | 207.44 | 206.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 206.70 | 207.44 | 206.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 13:15:00 | 207.70 | 207.65 | 206.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 15:15:00 | 208.65 | 207.72 | 207.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 12:15:00 | 208.85 | 208.55 | 207.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 10:00:00 | 208.90 | 208.84 | 208.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 10:30:00 | 208.95 | 208.85 | 208.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 12:15:00 | 209.00 | 208.87 | 208.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 12:30:00 | 208.90 | 208.87 | 208.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 208.75 | 208.86 | 208.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 14:30:00 | 208.40 | 208.86 | 208.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 208.20 | 208.73 | 208.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:15:00 | 208.10 | 208.73 | 208.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 209.80 | 208.94 | 208.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-05-30 14:15:00 | 207.50 | 208.40 | 208.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 207.50 | 208.40 | 208.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 15:15:00 | 207.00 | 208.12 | 208.31 | Break + close below crossover candle low |

### Cycle 6 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 209.80 | 208.46 | 208.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 10:15:00 | 210.15 | 208.80 | 208.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 15:15:00 | 211.20 | 211.51 | 210.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:15:00 | 212.40 | 211.51 | 210.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 210.70 | 211.35 | 210.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:00:00 | 210.70 | 211.35 | 210.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 211.80 | 211.44 | 210.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-02 11:15:00 | 210.25 | 211.20 | 210.72 | SL hit (close<ema400) qty=1.00 sl=210.72 alert=retest1 |

### Cycle 7 — SELL (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 10:15:00 | 210.90 | 211.19 | 211.20 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 15:15:00 | 212.00 | 211.28 | 211.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 09:15:00 | 212.45 | 211.85 | 211.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 210.50 | 211.61 | 211.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 210.50 | 211.61 | 211.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 210.50 | 211.61 | 211.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 210.50 | 211.61 | 211.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 210.40 | 211.37 | 211.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 13:15:00 | 209.95 | 211.08 | 211.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 206.15 | 206.01 | 207.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 11:15:00 | 206.95 | 206.28 | 207.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 11:15:00 | 206.95 | 206.28 | 207.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 11:30:00 | 207.30 | 206.28 | 207.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 206.75 | 206.37 | 206.86 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 11:15:00 | 209.25 | 207.20 | 207.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 14:15:00 | 210.15 | 208.20 | 207.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 11:15:00 | 209.00 | 209.07 | 208.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 11:45:00 | 208.75 | 209.07 | 208.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 207.70 | 208.80 | 208.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:45:00 | 207.95 | 208.80 | 208.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 207.55 | 208.55 | 208.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:00:00 | 207.55 | 208.55 | 208.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 207.35 | 208.31 | 208.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:45:00 | 207.10 | 208.31 | 208.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 213.70 | 214.28 | 212.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:30:00 | 212.85 | 214.28 | 212.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 219.90 | 215.06 | 213.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 14:15:00 | 221.00 | 217.01 | 214.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 15:15:00 | 220.80 | 217.61 | 215.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-05 10:15:00 | 243.10 | 240.52 | 238.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 11:15:00 | 250.75 | 252.66 | 252.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 14:15:00 | 248.35 | 251.11 | 252.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-18 09:15:00 | 250.85 | 250.78 | 251.69 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 11:15:00 | 248.00 | 250.30 | 251.39 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 13:45:00 | 247.85 | 248.46 | 250.18 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 248.15 | 248.19 | 249.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-20 09:15:00 | 249.40 | 248.37 | 248.93 | SL hit (close>ema400) qty=1.00 sl=248.93 alert=retest1 |

### Cycle 12 — BUY (started 2023-07-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 14:15:00 | 251.10 | 249.11 | 249.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 252.85 | 250.14 | 249.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 11:15:00 | 251.65 | 252.05 | 251.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 11:15:00 | 251.65 | 252.05 | 251.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 251.65 | 252.05 | 251.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 12:00:00 | 251.65 | 252.05 | 251.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 251.65 | 251.97 | 251.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:15:00 | 251.50 | 251.97 | 251.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 250.15 | 251.60 | 251.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 14:00:00 | 250.15 | 251.60 | 251.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 251.40 | 251.56 | 251.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 15:15:00 | 251.90 | 251.56 | 251.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 09:30:00 | 251.80 | 251.50 | 251.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 12:15:00 | 249.95 | 250.96 | 251.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 12:15:00 | 249.95 | 250.96 | 251.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 13:15:00 | 249.60 | 250.69 | 250.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 251.15 | 250.67 | 250.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 251.15 | 250.67 | 250.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 251.15 | 250.67 | 250.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:00:00 | 251.15 | 250.67 | 250.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 251.25 | 250.79 | 250.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:00:00 | 251.25 | 250.79 | 250.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 251.40 | 250.91 | 250.91 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 12:15:00 | 251.65 | 251.06 | 250.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 14:15:00 | 252.25 | 251.39 | 251.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 09:15:00 | 252.80 | 254.31 | 253.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 09:15:00 | 252.80 | 254.31 | 253.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 252.80 | 254.31 | 253.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:00:00 | 252.80 | 254.31 | 253.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 255.50 | 254.55 | 253.41 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 15:15:00 | 251.35 | 252.74 | 252.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-31 09:15:00 | 248.75 | 251.94 | 252.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 15:15:00 | 251.00 | 250.05 | 251.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 15:15:00 | 251.00 | 250.05 | 251.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 251.00 | 250.05 | 251.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:15:00 | 255.80 | 250.05 | 251.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 256.25 | 251.29 | 251.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:45:00 | 258.00 | 251.29 | 251.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 10:15:00 | 260.60 | 253.15 | 252.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 11:15:00 | 264.25 | 255.37 | 253.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 11:15:00 | 260.00 | 260.73 | 257.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 11:30:00 | 260.20 | 260.73 | 257.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 255.90 | 259.49 | 257.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:30:00 | 256.40 | 259.49 | 257.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 259.95 | 259.58 | 257.95 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 253.85 | 257.12 | 257.33 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 12:15:00 | 258.05 | 257.24 | 257.16 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 14:15:00 | 256.45 | 257.03 | 257.08 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 259.25 | 257.51 | 257.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 10:15:00 | 260.00 | 258.01 | 257.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 259.40 | 260.66 | 259.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 10:15:00 | 259.40 | 260.66 | 259.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 259.40 | 260.66 | 259.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:00:00 | 259.40 | 260.66 | 259.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 258.80 | 260.29 | 259.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 257.70 | 260.29 | 259.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 259.55 | 260.14 | 259.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 13:30:00 | 260.35 | 260.31 | 259.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 10:15:00 | 261.15 | 260.47 | 259.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 11:15:00 | 263.00 | 264.06 | 264.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 11:15:00 | 263.00 | 264.06 | 264.13 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 14:15:00 | 266.55 | 264.48 | 264.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 13:15:00 | 268.65 | 266.87 | 266.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 09:15:00 | 265.65 | 266.79 | 266.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 265.65 | 266.79 | 266.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 265.65 | 266.79 | 266.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:00:00 | 265.65 | 266.79 | 266.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 265.00 | 266.43 | 266.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:45:00 | 264.50 | 266.43 | 266.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 13:15:00 | 264.80 | 265.79 | 265.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 14:15:00 | 264.45 | 265.52 | 265.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 09:15:00 | 265.50 | 265.39 | 265.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 09:15:00 | 265.50 | 265.39 | 265.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 265.50 | 265.39 | 265.65 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 12:15:00 | 266.90 | 265.88 | 265.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 13:15:00 | 268.70 | 266.44 | 266.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 14:15:00 | 267.90 | 268.03 | 267.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 15:00:00 | 267.90 | 268.03 | 267.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 267.55 | 268.24 | 267.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:00:00 | 267.55 | 268.24 | 267.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 266.65 | 267.92 | 267.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 12:00:00 | 266.65 | 267.92 | 267.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 266.75 | 267.69 | 267.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:15:00 | 266.40 | 267.69 | 267.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 268.00 | 267.75 | 267.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:30:00 | 266.65 | 267.75 | 267.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 265.30 | 267.26 | 267.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 15:15:00 | 265.00 | 266.81 | 267.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 263.45 | 262.27 | 264.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 10:00:00 | 263.45 | 262.27 | 264.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 264.00 | 262.61 | 264.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:00:00 | 264.00 | 262.61 | 264.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 260.00 | 262.09 | 263.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 12:15:00 | 259.85 | 262.09 | 263.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:45:00 | 259.90 | 261.22 | 262.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 09:15:00 | 266.00 | 262.48 | 262.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 09:15:00 | 266.00 | 262.48 | 262.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 14:15:00 | 266.90 | 264.72 | 263.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 11:15:00 | 265.00 | 265.47 | 264.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-01 11:45:00 | 264.85 | 265.47 | 264.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 264.80 | 265.34 | 264.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:30:00 | 264.50 | 265.34 | 264.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 270.00 | 271.52 | 270.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 269.80 | 271.52 | 270.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 271.00 | 271.42 | 270.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 269.75 | 271.42 | 270.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 270.80 | 271.29 | 270.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:45:00 | 271.65 | 271.56 | 270.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 269.80 | 275.02 | 274.40 | SL hit (close<static) qty=1.00 sl=270.10 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 267.10 | 272.69 | 273.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 266.20 | 269.96 | 271.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 268.30 | 267.99 | 269.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 268.30 | 267.99 | 269.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 268.30 | 267.99 | 269.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 269.50 | 267.99 | 269.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 269.40 | 268.27 | 269.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 269.00 | 268.27 | 269.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 268.35 | 268.29 | 269.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:15:00 | 267.30 | 268.29 | 269.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 10:15:00 | 267.75 | 268.57 | 269.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 15:15:00 | 269.70 | 267.77 | 268.25 | SL hit (close>static) qty=1.00 sl=269.40 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 260.30 | 258.52 | 258.39 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 258.25 | 258.55 | 258.55 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 260.95 | 259.01 | 258.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 12:15:00 | 262.55 | 260.28 | 259.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 14:15:00 | 260.50 | 260.57 | 259.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-29 15:00:00 | 260.50 | 260.57 | 259.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 258.80 | 260.34 | 259.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:45:00 | 258.60 | 260.34 | 259.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 260.00 | 260.27 | 259.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:15:00 | 258.80 | 260.27 | 259.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 258.75 | 259.96 | 259.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 12:00:00 | 258.75 | 259.96 | 259.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 12:15:00 | 259.95 | 259.96 | 259.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 13:30:00 | 260.00 | 259.93 | 259.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 14:15:00 | 260.05 | 259.93 | 259.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 258.70 | 259.64 | 259.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 258.70 | 259.64 | 259.65 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 10:15:00 | 259.75 | 259.66 | 259.66 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 257.80 | 259.29 | 259.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 256.55 | 258.74 | 259.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 258.45 | 257.85 | 258.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 258.45 | 257.85 | 258.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 258.45 | 257.85 | 258.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 257.65 | 257.85 | 258.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 09:30:00 | 257.85 | 257.51 | 257.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 11:30:00 | 257.85 | 258.03 | 258.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 12:15:00 | 259.20 | 258.26 | 258.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 259.20 | 258.26 | 258.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 13:15:00 | 259.40 | 258.49 | 258.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 257.25 | 258.37 | 258.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 257.25 | 258.37 | 258.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 257.25 | 258.37 | 258.34 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 256.90 | 258.08 | 258.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 11:15:00 | 255.75 | 257.61 | 257.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 259.40 | 256.59 | 257.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 259.40 | 256.59 | 257.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 259.40 | 256.59 | 257.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:00:00 | 259.40 | 256.59 | 257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 258.20 | 256.91 | 257.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 12:15:00 | 257.75 | 257.10 | 257.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 14:15:00 | 258.40 | 257.60 | 257.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 14:15:00 | 258.40 | 257.60 | 257.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 260.50 | 258.34 | 257.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 263.95 | 264.08 | 262.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 11:00:00 | 263.95 | 264.08 | 262.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 264.00 | 264.06 | 262.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 11:45:00 | 262.90 | 264.06 | 262.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 263.60 | 263.97 | 262.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 10:45:00 | 264.10 | 263.14 | 262.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 11:30:00 | 264.45 | 263.35 | 262.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 12:00:00 | 264.15 | 263.35 | 262.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 13:00:00 | 264.10 | 263.50 | 263.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 266.30 | 267.79 | 266.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 266.30 | 267.79 | 266.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 265.20 | 267.27 | 266.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:00:00 | 265.20 | 267.27 | 266.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 264.60 | 266.74 | 266.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:45:00 | 264.10 | 266.74 | 266.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 267.10 | 266.86 | 266.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-19 10:15:00 | 264.70 | 266.12 | 266.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 264.70 | 266.12 | 266.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 11:15:00 | 263.95 | 265.69 | 265.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 15:15:00 | 265.50 | 265.31 | 265.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 15:15:00 | 265.50 | 265.31 | 265.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 265.50 | 265.31 | 265.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 264.40 | 265.31 | 265.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 15:15:00 | 251.18 | 254.61 | 258.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 249.80 | 247.08 | 249.78 | SL hit (close>ema200) qty=0.50 sl=247.08 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 255.00 | 251.22 | 250.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 255.85 | 253.37 | 252.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 254.45 | 254.90 | 253.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 10:00:00 | 254.45 | 254.90 | 253.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 253.80 | 254.68 | 253.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 253.80 | 254.68 | 253.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 254.10 | 254.56 | 253.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 12:15:00 | 254.70 | 254.56 | 253.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 12:15:00 | 253.25 | 254.30 | 253.67 | SL hit (close<static) qty=1.00 sl=253.45 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 15:15:00 | 281.50 | 282.66 | 282.80 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 284.30 | 282.81 | 282.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 286.00 | 283.56 | 283.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 12:15:00 | 283.75 | 283.85 | 283.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 12:15:00 | 283.75 | 283.85 | 283.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 283.75 | 283.85 | 283.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 14:15:00 | 284.95 | 283.92 | 283.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 09:15:00 | 282.30 | 283.96 | 283.64 | SL hit (close<static) qty=1.00 sl=283.40 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 289.50 | 292.83 | 292.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 13:15:00 | 287.95 | 291.85 | 292.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 293.30 | 291.61 | 292.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 293.30 | 291.61 | 292.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 293.30 | 291.61 | 292.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:00:00 | 293.30 | 291.61 | 292.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 293.50 | 291.99 | 292.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 11:00:00 | 293.50 | 291.99 | 292.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 11:15:00 | 293.60 | 292.31 | 292.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 11:30:00 | 294.25 | 292.31 | 292.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 12:15:00 | 292.80 | 292.41 | 292.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 15:15:00 | 293.95 | 293.02 | 292.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 09:15:00 | 292.95 | 293.00 | 292.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 292.95 | 293.00 | 292.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 292.95 | 293.00 | 292.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:00:00 | 292.95 | 293.00 | 292.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 292.40 | 292.88 | 292.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:00:00 | 292.40 | 292.88 | 292.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 11:15:00 | 292.20 | 292.75 | 292.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:45:00 | 292.15 | 292.75 | 292.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 12:15:00 | 291.40 | 292.48 | 292.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 14:15:00 | 288.60 | 291.46 | 292.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 288.70 | 288.10 | 289.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 288.70 | 288.10 | 289.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 288.70 | 288.10 | 289.49 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 14:15:00 | 295.90 | 290.39 | 290.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 11:15:00 | 300.25 | 295.95 | 294.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 09:15:00 | 304.65 | 304.93 | 301.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 10:00:00 | 304.65 | 304.93 | 301.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 303.55 | 304.54 | 302.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:45:00 | 302.95 | 304.54 | 302.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 299.10 | 303.45 | 301.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 299.10 | 303.45 | 301.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 291.95 | 301.15 | 300.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 293.30 | 301.15 | 300.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 287.65 | 298.45 | 299.74 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 14:15:00 | 296.00 | 295.24 | 295.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 298.35 | 296.06 | 295.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 316.30 | 319.48 | 315.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:30:00 | 316.90 | 319.48 | 315.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 323.60 | 324.46 | 322.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:45:00 | 323.45 | 324.46 | 322.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 326.55 | 324.88 | 323.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:45:00 | 323.80 | 324.88 | 323.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 323.75 | 324.86 | 323.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 323.75 | 324.86 | 323.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 329.05 | 325.69 | 323.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 10:45:00 | 331.10 | 328.24 | 327.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 13:15:00 | 324.85 | 331.77 | 332.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 13:15:00 | 324.85 | 331.77 | 332.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 320.15 | 325.57 | 328.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 319.90 | 319.54 | 322.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 12:00:00 | 319.90 | 319.54 | 322.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 321.85 | 320.14 | 321.85 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 13:15:00 | 324.95 | 322.64 | 322.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 327.30 | 323.57 | 323.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 10:15:00 | 324.30 | 324.39 | 323.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 10:15:00 | 324.30 | 324.39 | 323.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 324.30 | 324.39 | 323.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 10:45:00 | 324.00 | 324.39 | 323.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 11:15:00 | 323.90 | 324.29 | 323.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 11:45:00 | 323.65 | 324.29 | 323.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 322.70 | 323.97 | 323.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 13:00:00 | 322.70 | 323.97 | 323.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 323.70 | 323.92 | 323.57 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 15:15:00 | 321.50 | 323.09 | 323.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 316.70 | 321.09 | 322.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 318.25 | 317.86 | 319.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 10:30:00 | 318.70 | 317.86 | 319.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 313.35 | 315.45 | 317.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:30:00 | 311.15 | 314.27 | 316.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 09:15:00 | 317.15 | 315.09 | 314.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 317.15 | 315.09 | 314.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 12:15:00 | 319.80 | 316.69 | 315.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 12:15:00 | 328.75 | 329.73 | 325.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 12:30:00 | 327.50 | 329.73 | 325.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 344.70 | 347.26 | 345.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:30:00 | 345.00 | 347.26 | 345.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 344.80 | 346.77 | 345.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 14:00:00 | 347.50 | 346.91 | 345.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 12:00:00 | 347.60 | 347.08 | 346.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 341.15 | 345.36 | 345.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 341.15 | 345.36 | 345.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 334.70 | 343.23 | 344.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 330.30 | 329.07 | 332.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 330.30 | 329.07 | 332.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 332.45 | 330.20 | 332.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:00:00 | 332.45 | 330.20 | 332.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 332.60 | 330.68 | 332.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 332.60 | 330.68 | 332.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 334.40 | 331.42 | 332.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:00:00 | 334.40 | 331.42 | 332.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 333.45 | 331.83 | 332.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:30:00 | 334.80 | 331.83 | 332.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 335.75 | 333.14 | 333.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 335.75 | 333.14 | 333.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 337.40 | 334.00 | 333.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 338.55 | 336.28 | 335.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 336.35 | 336.60 | 335.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 15:00:00 | 336.35 | 336.60 | 335.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 336.60 | 336.60 | 335.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 340.80 | 336.60 | 335.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 15:15:00 | 337.25 | 337.87 | 336.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-20 09:15:00 | 330.90 | 336.38 | 336.38 | SL hit (close<static) qty=1.00 sl=335.40 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 328.00 | 334.70 | 335.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 15:15:00 | 326.85 | 329.25 | 331.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 330.25 | 328.88 | 330.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 11:15:00 | 330.25 | 328.88 | 330.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 330.25 | 328.88 | 330.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:45:00 | 330.95 | 328.88 | 330.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 331.40 | 329.38 | 330.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:45:00 | 331.15 | 329.38 | 330.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 331.40 | 329.78 | 330.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 331.40 | 329.78 | 330.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 332.35 | 330.30 | 330.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:30:00 | 332.70 | 330.30 | 330.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 331.75 | 331.24 | 331.21 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 13:15:00 | 330.65 | 331.16 | 331.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 15:15:00 | 329.65 | 330.68 | 330.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 10:15:00 | 330.65 | 330.47 | 330.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 10:15:00 | 330.65 | 330.47 | 330.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 330.65 | 330.47 | 330.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:00:00 | 330.65 | 330.47 | 330.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 329.70 | 330.32 | 330.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:30:00 | 330.85 | 330.32 | 330.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 329.10 | 330.07 | 330.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 12:30:00 | 330.45 | 330.07 | 330.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 324.95 | 327.02 | 328.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:15:00 | 324.45 | 327.02 | 328.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:45:00 | 321.85 | 326.00 | 327.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 09:30:00 | 324.25 | 320.64 | 322.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 14:15:00 | 323.95 | 322.45 | 322.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 14:15:00 | 323.70 | 322.70 | 322.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 14:15:00 | 323.70 | 322.70 | 322.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 327.00 | 323.93 | 323.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 10:15:00 | 326.60 | 327.77 | 326.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 10:15:00 | 326.60 | 327.77 | 326.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 326.60 | 327.77 | 326.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:00:00 | 326.60 | 327.77 | 326.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 325.85 | 327.38 | 326.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 325.85 | 327.38 | 326.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 325.50 | 327.01 | 326.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:30:00 | 324.75 | 327.01 | 326.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 325.00 | 326.60 | 326.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:45:00 | 324.40 | 326.60 | 326.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 326.90 | 326.44 | 326.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:15:00 | 324.40 | 326.44 | 326.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 319.60 | 325.07 | 325.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 10:15:00 | 317.80 | 323.62 | 324.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 321.85 | 321.82 | 323.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 14:30:00 | 322.25 | 321.82 | 323.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 324.30 | 322.27 | 323.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 324.30 | 322.27 | 323.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 326.85 | 323.18 | 323.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:00:00 | 326.85 | 323.18 | 323.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 327.90 | 324.13 | 324.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 13:15:00 | 329.50 | 326.02 | 324.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 325.80 | 326.50 | 325.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 10:00:00 | 325.80 | 326.50 | 325.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 323.80 | 325.96 | 325.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 11:00:00 | 323.80 | 325.96 | 325.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 322.50 | 325.27 | 325.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 12:00:00 | 322.50 | 325.27 | 325.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 12:15:00 | 323.65 | 324.95 | 324.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 322.10 | 324.00 | 324.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 308.55 | 307.43 | 312.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 309.35 | 307.43 | 312.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 307.00 | 304.91 | 306.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:30:00 | 306.60 | 304.91 | 306.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 305.05 | 304.94 | 306.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:30:00 | 308.60 | 304.94 | 306.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 306.95 | 305.34 | 306.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:00:00 | 306.95 | 305.34 | 306.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 307.40 | 305.75 | 306.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:15:00 | 301.60 | 306.45 | 306.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 306.15 | 301.79 | 301.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 306.15 | 301.79 | 301.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 13:15:00 | 307.55 | 305.57 | 304.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 305.00 | 305.90 | 304.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 305.00 | 305.90 | 304.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 305.00 | 305.90 | 304.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 304.60 | 305.90 | 304.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 305.90 | 305.90 | 304.74 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 10:15:00 | 304.15 | 304.65 | 304.68 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 14:15:00 | 305.55 | 304.80 | 304.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 15:15:00 | 306.00 | 305.04 | 304.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 14:15:00 | 313.70 | 314.09 | 312.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 15:00:00 | 313.70 | 314.09 | 312.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 452.75 | 453.27 | 440.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 11:45:00 | 468.70 | 453.80 | 446.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 11:30:00 | 464.60 | 463.57 | 456.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 13:30:00 | 464.75 | 464.12 | 457.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 14:00:00 | 466.20 | 464.12 | 457.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 445.05 | 459.33 | 457.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-24 11:15:00 | 445.45 | 454.50 | 455.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 11:15:00 | 445.45 | 454.50 | 455.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 09:15:00 | 442.65 | 447.79 | 451.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 13:15:00 | 444.45 | 443.45 | 447.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 14:00:00 | 444.45 | 443.45 | 447.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 447.15 | 444.19 | 447.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 15:00:00 | 447.15 | 444.19 | 447.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 446.60 | 444.67 | 447.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:15:00 | 449.50 | 444.67 | 447.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 446.70 | 445.08 | 447.42 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 12:15:00 | 452.35 | 449.19 | 448.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 13:15:00 | 468.65 | 453.08 | 450.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 457.25 | 457.52 | 453.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 10:00:00 | 457.25 | 457.52 | 453.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 455.60 | 457.16 | 454.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:30:00 | 455.90 | 457.16 | 454.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 454.55 | 456.63 | 454.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 13:45:00 | 455.00 | 456.63 | 454.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 455.15 | 456.34 | 454.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 460.85 | 456.17 | 454.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 10:15:00 | 459.45 | 463.78 | 463.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 11:15:00 | 459.65 | 462.50 | 462.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 11:15:00 | 459.65 | 462.50 | 462.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 458.05 | 460.94 | 461.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 10:15:00 | 458.90 | 457.54 | 459.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 10:15:00 | 458.90 | 457.54 | 459.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 458.90 | 457.54 | 459.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 458.90 | 457.54 | 459.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 467.35 | 459.50 | 460.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 467.35 | 459.50 | 460.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 12:15:00 | 465.30 | 460.66 | 460.53 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 09:15:00 | 455.10 | 459.74 | 460.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 454.60 | 458.71 | 459.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 452.90 | 452.71 | 455.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 09:30:00 | 453.75 | 452.71 | 455.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 453.30 | 452.83 | 455.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:30:00 | 454.25 | 452.83 | 455.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 453.30 | 452.93 | 455.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:30:00 | 456.30 | 452.93 | 455.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 451.95 | 448.58 | 451.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:00:00 | 451.95 | 448.58 | 451.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 456.40 | 450.15 | 451.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:00:00 | 456.40 | 450.15 | 451.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 455.25 | 451.17 | 452.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:30:00 | 452.95 | 452.00 | 452.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 458.60 | 453.32 | 452.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 458.60 | 453.32 | 452.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 460.45 | 455.51 | 454.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 457.80 | 457.84 | 456.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 12:00:00 | 457.80 | 457.84 | 456.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 456.00 | 457.48 | 456.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:00:00 | 456.00 | 457.48 | 456.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 454.75 | 456.93 | 455.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:00:00 | 454.75 | 456.93 | 455.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 463.45 | 458.23 | 456.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 10:30:00 | 470.80 | 462.56 | 459.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 471.65 | 475.67 | 474.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 12:15:00 | 467.50 | 472.89 | 473.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 467.50 | 472.89 | 473.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 465.60 | 471.43 | 472.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 10:15:00 | 474.80 | 467.25 | 468.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 10:15:00 | 474.80 | 467.25 | 468.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 474.80 | 467.25 | 468.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 474.80 | 467.25 | 468.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 11:15:00 | 484.75 | 470.75 | 470.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 10:15:00 | 491.50 | 479.54 | 475.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 10:15:00 | 505.85 | 506.61 | 498.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:45:00 | 505.25 | 506.61 | 498.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 500.05 | 505.91 | 501.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 500.05 | 505.91 | 501.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 501.00 | 504.93 | 501.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:15:00 | 501.00 | 504.93 | 501.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 500.40 | 504.02 | 501.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:00:00 | 500.40 | 504.02 | 501.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 500.75 | 503.37 | 501.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 508.40 | 501.22 | 500.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 498.20 | 500.52 | 500.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 498.20 | 500.52 | 500.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 14:15:00 | 488.95 | 497.78 | 499.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 501.85 | 497.28 | 498.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 501.85 | 497.28 | 498.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 501.85 | 497.28 | 498.74 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 504.25 | 499.82 | 499.62 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 491.00 | 499.97 | 499.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 456.10 | 485.07 | 492.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 488.80 | 473.35 | 481.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 488.80 | 473.35 | 481.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 488.80 | 473.35 | 481.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 488.80 | 473.35 | 481.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 488.70 | 476.42 | 482.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 492.20 | 476.42 | 482.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 519.85 | 491.21 | 487.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 528.15 | 511.94 | 501.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 526.00 | 527.04 | 518.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:00:00 | 526.00 | 527.04 | 518.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 558.60 | 557.48 | 551.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 553.30 | 557.48 | 551.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 566.95 | 569.49 | 566.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:45:00 | 569.70 | 569.19 | 567.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 13:00:00 | 569.90 | 569.33 | 567.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 15:00:00 | 574.05 | 570.20 | 568.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 568.00 | 575.30 | 575.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 568.00 | 575.30 | 575.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 564.55 | 571.35 | 573.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 568.10 | 567.13 | 570.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 568.10 | 567.13 | 570.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 572.00 | 567.12 | 569.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 572.00 | 567.12 | 569.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 570.40 | 567.78 | 569.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:00:00 | 564.10 | 567.45 | 569.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:00:00 | 564.80 | 566.39 | 568.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 571.60 | 569.04 | 568.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 571.60 | 569.04 | 568.85 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 565.05 | 568.24 | 568.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 564.30 | 567.45 | 568.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 11:15:00 | 566.10 | 564.84 | 566.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 11:15:00 | 566.10 | 564.84 | 566.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 566.10 | 564.84 | 566.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:00:00 | 566.10 | 564.84 | 566.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 566.60 | 565.19 | 566.35 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 15:15:00 | 568.40 | 566.98 | 566.98 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 566.45 | 566.88 | 566.94 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 568.15 | 567.01 | 566.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 15:15:00 | 572.20 | 569.61 | 568.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 13:15:00 | 572.05 | 572.24 | 570.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 14:00:00 | 572.05 | 572.24 | 570.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 563.95 | 572.29 | 571.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 563.95 | 572.29 | 571.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 560.50 | 569.93 | 570.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 548.05 | 552.13 | 555.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 547.20 | 546.81 | 550.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 12:45:00 | 546.65 | 546.81 | 550.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 547.65 | 547.07 | 549.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 548.40 | 547.07 | 549.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 550.30 | 547.72 | 549.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 550.30 | 547.72 | 549.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 551.85 | 548.54 | 549.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 551.85 | 548.54 | 549.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 544.50 | 547.73 | 549.20 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 551.40 | 544.94 | 544.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 554.05 | 546.76 | 545.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 552.60 | 553.05 | 549.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 552.60 | 553.05 | 549.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 553.00 | 553.04 | 549.93 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 543.75 | 547.77 | 548.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 13:15:00 | 540.50 | 546.32 | 547.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 495.30 | 493.37 | 503.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 495.30 | 493.37 | 503.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 496.00 | 490.18 | 493.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 496.00 | 490.18 | 493.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 498.00 | 491.75 | 493.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 492.95 | 491.75 | 493.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 11:15:00 | 499.50 | 491.40 | 491.88 | SL hit (close>static) qty=1.00 sl=498.90 alert=retest2 |

### Cycle 84 — BUY (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 12:15:00 | 496.20 | 492.36 | 492.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 504.50 | 498.24 | 495.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 498.75 | 500.29 | 497.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 12:15:00 | 498.75 | 500.29 | 497.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 498.75 | 500.29 | 497.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 499.90 | 500.29 | 497.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 496.05 | 499.44 | 497.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 496.05 | 499.44 | 497.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 492.20 | 497.99 | 496.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 492.20 | 497.99 | 496.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 488.85 | 495.61 | 495.98 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 496.00 | 493.47 | 493.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 498.30 | 494.67 | 493.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 495.10 | 495.88 | 494.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 495.10 | 495.88 | 494.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 495.10 | 495.88 | 494.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:45:00 | 494.55 | 495.88 | 494.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 493.10 | 495.33 | 494.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 493.00 | 495.33 | 494.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 492.15 | 494.69 | 494.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 491.40 | 494.69 | 494.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 497.40 | 495.21 | 494.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 509.05 | 495.90 | 495.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 15:15:00 | 501.00 | 504.91 | 505.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 501.00 | 504.91 | 505.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 11:15:00 | 498.05 | 501.89 | 503.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 502.75 | 500.03 | 501.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 502.75 | 500.03 | 501.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 502.75 | 500.03 | 501.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 502.20 | 500.03 | 501.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 501.20 | 500.26 | 501.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 502.35 | 500.26 | 501.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 500.00 | 500.21 | 501.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:45:00 | 498.05 | 500.02 | 501.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 498.65 | 499.96 | 501.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 13:15:00 | 498.05 | 489.77 | 489.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 13:15:00 | 498.05 | 489.77 | 489.00 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 486.90 | 489.03 | 489.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 485.45 | 488.31 | 488.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 476.20 | 475.45 | 480.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 482.80 | 475.45 | 480.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 477.25 | 475.81 | 479.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 481.35 | 475.81 | 479.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 480.65 | 477.40 | 479.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:00:00 | 480.65 | 477.40 | 479.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 479.85 | 477.89 | 479.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 476.90 | 478.57 | 479.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:30:00 | 476.70 | 476.44 | 477.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 481.55 | 478.68 | 478.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 481.55 | 478.68 | 478.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 11:15:00 | 484.05 | 479.76 | 478.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 483.00 | 483.65 | 481.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:45:00 | 483.35 | 483.65 | 481.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 484.50 | 486.04 | 484.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 485.35 | 486.04 | 484.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 484.30 | 485.69 | 484.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:45:00 | 484.90 | 485.69 | 484.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 484.50 | 485.45 | 484.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:30:00 | 484.40 | 485.45 | 484.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 484.25 | 485.21 | 484.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:15:00 | 483.65 | 485.21 | 484.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 482.30 | 484.63 | 484.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 481.80 | 484.63 | 484.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 482.95 | 484.29 | 483.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 481.15 | 484.29 | 483.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 479.70 | 483.38 | 483.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 475.60 | 481.82 | 482.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 462.95 | 462.90 | 468.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 14:15:00 | 465.70 | 464.32 | 467.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 465.70 | 464.32 | 467.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:45:00 | 466.40 | 464.32 | 467.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 467.20 | 465.29 | 467.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 468.05 | 465.29 | 467.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 468.30 | 465.89 | 467.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:30:00 | 468.90 | 465.89 | 467.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 469.55 | 466.62 | 467.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:30:00 | 469.80 | 466.62 | 467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 471.45 | 468.38 | 468.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 475.15 | 470.49 | 469.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 473.50 | 474.34 | 472.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 473.50 | 474.34 | 472.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 470.25 | 473.48 | 472.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 470.25 | 473.48 | 472.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 469.55 | 472.69 | 471.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 469.55 | 472.69 | 471.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 468.75 | 472.02 | 471.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 468.30 | 472.02 | 471.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 469.45 | 471.50 | 471.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 467.45 | 470.28 | 471.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 471.95 | 470.47 | 471.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 471.95 | 470.47 | 471.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 471.95 | 470.47 | 471.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 471.95 | 470.47 | 471.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 471.70 | 470.71 | 471.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 474.00 | 470.71 | 471.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 480.80 | 472.73 | 471.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 487.60 | 475.70 | 473.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 506.35 | 507.73 | 500.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 506.35 | 507.73 | 500.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 506.35 | 507.73 | 500.82 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 486.95 | 500.26 | 500.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 483.20 | 496.85 | 499.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 491.60 | 487.49 | 491.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 491.60 | 487.49 | 491.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 491.60 | 487.49 | 491.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 491.60 | 487.49 | 491.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 490.20 | 488.03 | 491.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 495.25 | 488.03 | 491.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 492.95 | 489.01 | 491.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 492.95 | 489.01 | 491.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 492.10 | 489.63 | 491.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 14:30:00 | 491.30 | 490.43 | 492.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 494.00 | 491.14 | 492.22 | SL hit (close>static) qty=1.00 sl=493.95 alert=retest2 |

### Cycle 96 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 500.65 | 493.04 | 492.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 518.80 | 498.19 | 495.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 509.35 | 514.20 | 508.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 509.35 | 514.20 | 508.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 509.35 | 514.20 | 508.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 509.35 | 514.20 | 508.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 505.60 | 512.48 | 507.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 505.60 | 512.48 | 507.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 512.45 | 512.47 | 508.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:15:00 | 513.00 | 512.47 | 508.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 516.50 | 521.18 | 521.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 516.50 | 521.18 | 521.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 512.65 | 519.48 | 520.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 472.20 | 472.11 | 481.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 472.20 | 472.11 | 481.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 462.75 | 455.75 | 461.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 462.75 | 455.75 | 461.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 465.35 | 457.67 | 461.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 463.30 | 457.67 | 461.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 456.80 | 460.94 | 462.09 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 467.00 | 463.08 | 462.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 469.65 | 465.67 | 464.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 467.30 | 467.54 | 465.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 467.30 | 467.54 | 465.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 465.20 | 467.07 | 465.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 465.20 | 467.07 | 465.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 466.00 | 466.86 | 465.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 457.95 | 466.86 | 465.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 457.55 | 465.00 | 464.80 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 457.50 | 463.50 | 464.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 451.90 | 460.02 | 462.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 462.75 | 458.63 | 460.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 462.75 | 458.63 | 460.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 462.75 | 458.63 | 460.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 462.75 | 458.63 | 460.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 457.35 | 458.37 | 460.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 449.95 | 458.37 | 460.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 14:15:00 | 451.85 | 448.70 | 448.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 451.85 | 448.70 | 448.59 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 444.80 | 448.72 | 449.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 442.55 | 447.49 | 448.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 441.40 | 441.03 | 444.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 441.40 | 441.03 | 444.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 442.55 | 439.55 | 441.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 442.55 | 439.55 | 441.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 443.25 | 440.29 | 442.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:45:00 | 444.50 | 440.29 | 442.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 438.95 | 440.02 | 441.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 443.35 | 440.02 | 441.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 428.90 | 419.38 | 421.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 427.85 | 419.38 | 421.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 428.15 | 421.14 | 422.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 428.80 | 421.14 | 422.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 426.85 | 423.20 | 422.97 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 419.45 | 422.61 | 422.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 413.95 | 420.88 | 422.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 420.20 | 417.76 | 419.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 420.20 | 417.76 | 419.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 420.20 | 417.76 | 419.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 420.20 | 417.76 | 419.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 418.85 | 417.98 | 419.38 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 423.35 | 420.10 | 420.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 424.50 | 420.98 | 420.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 428.50 | 428.95 | 426.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 428.50 | 428.95 | 426.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 427.00 | 428.56 | 426.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:00:00 | 427.00 | 428.56 | 426.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 427.15 | 428.28 | 426.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:30:00 | 427.20 | 428.28 | 426.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 425.70 | 427.76 | 426.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 425.70 | 427.76 | 426.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 425.55 | 427.32 | 426.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 430.45 | 427.32 | 426.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 13:15:00 | 462.50 | 465.03 | 465.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 462.50 | 465.03 | 465.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 461.10 | 463.77 | 464.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 14:15:00 | 457.25 | 455.11 | 457.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 14:15:00 | 457.25 | 455.11 | 457.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 457.25 | 455.11 | 457.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 457.25 | 455.11 | 457.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 462.20 | 456.53 | 457.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 463.75 | 456.53 | 457.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 462.20 | 457.66 | 458.07 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 461.60 | 458.45 | 458.39 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 455.50 | 457.86 | 458.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 454.00 | 456.74 | 457.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 442.00 | 441.90 | 445.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 442.00 | 441.90 | 445.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 442.00 | 441.90 | 445.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:30:00 | 440.40 | 441.66 | 445.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 439.70 | 441.66 | 445.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 418.38 | 433.47 | 439.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 417.71 | 433.47 | 439.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 12:15:00 | 417.90 | 417.87 | 422.03 | SL hit (close>ema200) qty=0.50 sl=417.87 alert=retest2 |

### Cycle 108 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 418.95 | 416.74 | 416.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 421.50 | 417.69 | 417.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 419.40 | 419.49 | 418.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 11:00:00 | 419.40 | 419.49 | 418.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 421.60 | 419.91 | 418.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:30:00 | 419.90 | 419.91 | 418.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 424.00 | 425.44 | 423.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 418.75 | 425.44 | 423.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 418.80 | 424.11 | 423.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 418.85 | 424.11 | 423.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 409.10 | 421.11 | 421.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 408.45 | 414.83 | 418.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 383.35 | 380.46 | 387.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 383.35 | 380.46 | 387.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 383.05 | 381.65 | 384.81 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 386.15 | 384.94 | 384.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 388.95 | 386.16 | 385.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 388.70 | 388.77 | 387.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 388.70 | 388.77 | 387.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 388.70 | 388.77 | 387.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 388.75 | 388.77 | 387.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 387.55 | 388.53 | 387.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 388.40 | 388.53 | 387.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 388.50 | 388.52 | 387.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 390.50 | 389.29 | 388.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 386.15 | 389.39 | 388.66 | SL hit (close<static) qty=1.00 sl=387.40 alert=retest2 |

### Cycle 111 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 383.25 | 388.16 | 388.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 382.15 | 386.31 | 387.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 376.35 | 375.25 | 379.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 376.35 | 375.25 | 379.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 382.90 | 376.86 | 379.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 382.90 | 376.86 | 379.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 384.50 | 378.39 | 380.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 384.70 | 378.39 | 380.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 384.05 | 381.41 | 381.11 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 377.65 | 380.51 | 380.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 374.70 | 378.76 | 379.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 359.80 | 349.13 | 355.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 359.80 | 349.13 | 355.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 359.80 | 349.13 | 355.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 359.80 | 349.13 | 355.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 354.70 | 350.24 | 355.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:00:00 | 352.90 | 350.77 | 354.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 15:15:00 | 352.10 | 351.06 | 353.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 11:15:00 | 360.25 | 355.77 | 355.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 360.25 | 355.77 | 355.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 368.80 | 360.63 | 358.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 375.05 | 379.73 | 373.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 375.05 | 379.73 | 373.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 375.05 | 379.73 | 373.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 375.05 | 379.73 | 373.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 377.20 | 379.22 | 374.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:30:00 | 373.25 | 379.22 | 374.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 376.85 | 378.75 | 374.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:30:00 | 374.80 | 378.75 | 374.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 373.40 | 377.68 | 374.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 373.40 | 377.68 | 374.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 370.95 | 376.33 | 374.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:45:00 | 370.80 | 376.33 | 374.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 373.50 | 375.43 | 373.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 380.40 | 375.43 | 373.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 15:15:00 | 378.95 | 379.85 | 379.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 15:15:00 | 378.95 | 379.85 | 379.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 375.90 | 379.06 | 379.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 363.60 | 362.60 | 366.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 364.90 | 362.60 | 366.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 364.70 | 363.56 | 366.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 365.90 | 363.56 | 366.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 369.00 | 365.35 | 366.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 368.80 | 365.35 | 366.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 368.45 | 365.97 | 366.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 368.80 | 365.97 | 366.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 367.20 | 366.46 | 366.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:00:00 | 366.75 | 366.52 | 366.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:30:00 | 366.75 | 366.62 | 366.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 364.30 | 366.69 | 366.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 348.41 | 358.87 | 362.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 348.41 | 358.87 | 362.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 11:15:00 | 358.00 | 357.72 | 361.02 | SL hit (close>ema200) qty=0.50 sl=357.72 alert=retest2 |

### Cycle 116 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 364.40 | 360.28 | 359.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 366.75 | 363.92 | 362.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 12:15:00 | 370.00 | 371.11 | 368.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:00:00 | 370.00 | 371.11 | 368.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 366.95 | 370.28 | 368.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 366.55 | 370.28 | 368.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 366.35 | 369.49 | 367.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 366.20 | 369.49 | 367.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 360.20 | 366.76 | 366.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 356.70 | 362.71 | 364.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 347.25 | 345.62 | 350.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 347.25 | 345.62 | 350.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 348.10 | 346.56 | 349.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 348.10 | 346.56 | 349.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 349.30 | 347.10 | 349.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 349.30 | 347.10 | 349.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 346.95 | 347.07 | 349.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:45:00 | 344.70 | 346.53 | 348.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 345.55 | 346.46 | 348.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 355.40 | 348.02 | 348.71 | SL hit (close>static) qty=1.00 sl=349.80 alert=retest2 |

### Cycle 118 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 355.55 | 349.52 | 349.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 356.95 | 351.01 | 350.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 354.90 | 354.96 | 352.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 354.90 | 354.96 | 352.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 356.20 | 355.75 | 354.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:30:00 | 359.10 | 355.12 | 354.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 351.90 | 354.62 | 354.42 | SL hit (close<static) qty=1.00 sl=352.20 alert=retest2 |

### Cycle 119 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 351.50 | 354.00 | 354.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 349.20 | 353.04 | 353.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 337.05 | 336.77 | 340.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 337.05 | 336.77 | 340.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 337.05 | 336.77 | 340.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 338.85 | 336.77 | 340.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 344.20 | 337.05 | 338.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 344.20 | 337.05 | 338.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 342.05 | 338.05 | 338.74 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 342.40 | 339.63 | 339.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 343.80 | 340.47 | 339.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 361.20 | 363.28 | 359.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 361.20 | 363.28 | 359.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 362.20 | 362.67 | 359.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 358.55 | 362.67 | 359.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 360.35 | 362.19 | 360.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 360.35 | 362.19 | 360.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 359.20 | 361.59 | 360.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 373.30 | 361.59 | 360.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 15:15:00 | 357.50 | 361.51 | 361.25 | SL hit (close<static) qty=1.00 sl=359.05 alert=retest2 |

### Cycle 121 — SELL (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 10:15:00 | 359.40 | 360.91 | 361.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 357.35 | 360.20 | 360.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 360.70 | 359.42 | 360.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 360.70 | 359.42 | 360.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 360.70 | 359.42 | 360.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 360.70 | 359.42 | 360.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 359.40 | 359.42 | 360.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 362.30 | 359.42 | 360.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 360.75 | 359.68 | 360.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 358.75 | 359.62 | 360.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 364.75 | 360.57 | 360.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 364.75 | 360.57 | 360.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 367.50 | 364.70 | 362.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 370.60 | 370.72 | 367.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 12:45:00 | 371.00 | 370.72 | 367.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 362.50 | 369.72 | 368.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 362.80 | 369.72 | 368.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 364.80 | 368.74 | 368.12 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 363.80 | 367.06 | 367.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 363.35 | 366.32 | 367.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 353.70 | 353.65 | 358.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 353.70 | 353.65 | 358.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 356.15 | 354.50 | 358.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 355.15 | 355.01 | 358.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 354.15 | 357.18 | 358.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 13:15:00 | 362.20 | 359.14 | 358.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 362.20 | 359.14 | 358.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 366.50 | 362.06 | 360.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 376.85 | 378.52 | 374.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 14:45:00 | 377.10 | 378.52 | 374.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 370.50 | 376.58 | 374.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 370.50 | 376.58 | 374.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 375.20 | 376.31 | 374.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 377.00 | 376.31 | 374.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 379.05 | 376.81 | 375.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 373.85 | 380.78 | 381.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 373.85 | 380.78 | 381.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 370.00 | 378.63 | 380.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 379.20 | 373.98 | 376.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 11:15:00 | 379.20 | 373.98 | 376.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 379.20 | 373.98 | 376.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 379.20 | 373.98 | 376.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 380.50 | 375.29 | 376.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 380.35 | 375.29 | 376.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 377.50 | 376.30 | 376.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 382.45 | 376.30 | 376.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 376.60 | 376.36 | 376.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 378.60 | 376.36 | 376.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 376.25 | 376.34 | 376.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 376.25 | 376.34 | 376.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 374.50 | 372.62 | 374.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 374.50 | 372.62 | 374.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 373.95 | 372.89 | 374.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:45:00 | 362.80 | 369.27 | 372.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 12:15:00 | 372.70 | 365.17 | 364.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 12:15:00 | 372.70 | 365.17 | 364.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 12:15:00 | 375.25 | 370.56 | 367.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 11:15:00 | 371.80 | 373.13 | 370.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 12:00:00 | 371.80 | 373.13 | 370.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 370.15 | 372.53 | 370.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 370.15 | 372.53 | 370.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 364.45 | 370.92 | 370.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 364.45 | 370.92 | 370.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 361.80 | 369.09 | 369.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 359.00 | 367.07 | 368.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 361.80 | 361.18 | 363.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 371.35 | 361.18 | 363.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 375.30 | 364.01 | 364.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 375.30 | 364.01 | 364.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 376.35 | 366.47 | 365.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 378.35 | 371.57 | 368.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 376.40 | 376.56 | 372.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 376.40 | 376.56 | 372.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 391.55 | 392.68 | 389.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 390.75 | 392.68 | 389.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 388.85 | 391.62 | 389.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 388.80 | 391.62 | 389.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 389.00 | 391.10 | 389.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 389.00 | 391.10 | 389.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 386.15 | 390.11 | 389.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 386.15 | 390.11 | 389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 383.00 | 388.69 | 388.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 382.10 | 384.03 | 385.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 385.45 | 383.44 | 384.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 385.45 | 383.44 | 384.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 385.45 | 383.44 | 384.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 385.45 | 383.44 | 384.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 383.65 | 383.48 | 384.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 382.80 | 383.39 | 384.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:30:00 | 382.65 | 383.12 | 384.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 14:15:00 | 387.05 | 384.51 | 384.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 387.05 | 384.51 | 384.34 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 382.75 | 384.26 | 384.33 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 385.40 | 384.50 | 384.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 386.75 | 385.19 | 384.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 386.10 | 386.23 | 385.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 14:00:00 | 386.10 | 386.23 | 385.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 386.95 | 386.37 | 385.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 388.05 | 386.69 | 385.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:45:00 | 388.90 | 387.13 | 386.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 387.90 | 386.97 | 386.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 389.95 | 387.47 | 387.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 388.25 | 387.63 | 387.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 387.80 | 387.63 | 387.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 387.45 | 387.59 | 387.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 387.45 | 387.59 | 387.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 387.00 | 387.47 | 387.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 390.65 | 387.47 | 387.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 387.75 | 388.03 | 387.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 385.70 | 387.56 | 387.47 | SL hit (close<static) qty=1.00 sl=386.05 alert=retest2 |

### Cycle 133 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 386.25 | 387.30 | 387.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 383.85 | 386.33 | 386.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 387.20 | 386.50 | 386.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 387.20 | 386.50 | 386.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 387.20 | 386.50 | 386.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 387.20 | 386.50 | 386.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 387.10 | 386.62 | 386.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 387.50 | 386.62 | 386.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 388.20 | 386.94 | 387.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 388.20 | 386.94 | 387.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 388.05 | 387.16 | 387.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 15:15:00 | 388.75 | 387.68 | 387.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 404.10 | 404.77 | 401.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:45:00 | 404.55 | 404.77 | 401.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 400.90 | 404.01 | 402.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 400.80 | 404.01 | 402.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 401.15 | 403.44 | 402.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 401.15 | 403.44 | 402.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 394.30 | 400.13 | 400.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 393.30 | 398.76 | 400.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 384.95 | 383.73 | 385.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:00:00 | 384.95 | 383.73 | 385.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 379.55 | 376.74 | 379.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 379.55 | 376.74 | 379.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 379.95 | 377.38 | 379.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 379.50 | 377.38 | 379.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 380.20 | 377.94 | 379.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 379.85 | 377.94 | 379.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 378.75 | 378.37 | 379.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 379.75 | 378.37 | 379.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 380.10 | 378.72 | 379.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 380.10 | 378.72 | 379.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 380.50 | 379.07 | 379.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 379.40 | 379.07 | 379.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:45:00 | 378.60 | 379.42 | 379.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:45:00 | 379.70 | 379.59 | 379.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 379.15 | 379.59 | 379.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 378.40 | 379.35 | 379.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 386.80 | 380.72 | 380.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 386.80 | 380.72 | 380.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 387.90 | 382.16 | 380.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 385.80 | 385.86 | 384.47 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:15:00 | 387.05 | 385.86 | 384.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 385.95 | 385.88 | 384.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 385.95 | 385.88 | 384.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 386.00 | 385.90 | 384.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:30:00 | 386.25 | 385.90 | 384.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 386.25 | 385.97 | 384.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:15:00 | 387.10 | 385.97 | 384.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 387.30 | 385.96 | 385.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 386.70 | 388.48 | 387.57 | SL hit (close<ema400) qty=1.00 sl=387.57 alert=retest1 |

### Cycle 137 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 385.75 | 387.67 | 387.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 383.65 | 386.01 | 386.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 12:15:00 | 383.40 | 383.30 | 384.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 12:30:00 | 383.35 | 383.30 | 384.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 383.50 | 383.33 | 384.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:45:00 | 384.00 | 383.33 | 384.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 385.90 | 383.47 | 384.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 386.60 | 383.47 | 384.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 386.40 | 384.06 | 384.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 386.40 | 384.06 | 384.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 384.50 | 383.72 | 384.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 383.95 | 383.72 | 384.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 384.45 | 383.87 | 384.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 383.30 | 383.87 | 384.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 384.90 | 384.08 | 384.18 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 385.45 | 384.35 | 384.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 13:15:00 | 386.00 | 384.79 | 384.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 384.95 | 385.21 | 384.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 384.95 | 385.21 | 384.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 384.95 | 385.21 | 384.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 384.95 | 385.21 | 384.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 383.35 | 384.84 | 384.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 383.35 | 384.84 | 384.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 385.30 | 384.93 | 384.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:45:00 | 386.00 | 385.15 | 384.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 383.95 | 386.47 | 386.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 383.95 | 386.47 | 386.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 382.70 | 385.22 | 386.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 383.00 | 382.20 | 383.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 14:15:00 | 383.00 | 382.20 | 383.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 383.00 | 382.20 | 383.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 383.00 | 382.20 | 383.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 383.55 | 382.47 | 383.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 386.30 | 382.47 | 383.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 385.50 | 383.07 | 383.75 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 386.20 | 384.31 | 384.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 386.65 | 384.97 | 384.54 | Break + close above crossover candle high |

### Cycle 141 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 380.45 | 384.30 | 384.33 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 385.65 | 384.29 | 384.23 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 383.80 | 384.20 | 384.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 380.80 | 383.14 | 383.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 10:15:00 | 383.55 | 383.22 | 383.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 11:00:00 | 383.55 | 383.22 | 383.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 384.10 | 383.40 | 383.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 384.65 | 383.40 | 383.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 383.90 | 383.50 | 383.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 383.90 | 383.50 | 383.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 382.50 | 383.30 | 383.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 383.20 | 383.30 | 383.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 384.10 | 383.46 | 383.65 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 388.20 | 384.53 | 384.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 390.05 | 386.10 | 384.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 393.35 | 393.39 | 390.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:00:00 | 393.35 | 393.39 | 390.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 391.30 | 392.58 | 391.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 391.30 | 392.58 | 391.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 391.40 | 392.33 | 391.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 394.90 | 392.33 | 391.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 394.90 | 392.84 | 391.57 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 386.00 | 391.27 | 391.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 383.85 | 389.79 | 390.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 11:15:00 | 384.20 | 383.72 | 386.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:30:00 | 384.55 | 383.72 | 386.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 390.35 | 384.43 | 385.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 390.35 | 384.43 | 385.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 389.15 | 385.37 | 385.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 389.40 | 385.37 | 385.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 391.10 | 386.52 | 386.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 391.60 | 387.53 | 386.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 11:15:00 | 389.65 | 390.13 | 388.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 12:00:00 | 389.65 | 390.13 | 388.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 386.75 | 389.73 | 389.10 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 386.70 | 388.59 | 388.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 384.65 | 387.55 | 388.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 383.85 | 382.20 | 384.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 383.85 | 382.20 | 384.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 383.85 | 382.20 | 384.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 384.30 | 382.20 | 384.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 384.85 | 382.73 | 384.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 381.55 | 382.87 | 384.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 387.35 | 383.77 | 384.51 | SL hit (close>static) qty=1.00 sl=386.90 alert=retest2 |

### Cycle 148 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 391.25 | 385.80 | 385.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 393.90 | 389.12 | 387.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 379.80 | 387.80 | 386.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 13:15:00 | 379.80 | 387.80 | 386.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 379.80 | 387.80 | 386.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 379.80 | 387.80 | 386.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 386.10 | 387.46 | 386.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 386.85 | 386.95 | 386.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 382.10 | 385.98 | 386.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 382.10 | 385.98 | 386.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 381.00 | 383.10 | 384.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 379.70 | 378.81 | 381.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 379.40 | 378.81 | 381.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 377.50 | 378.66 | 380.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 375.80 | 378.11 | 380.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 376.00 | 377.69 | 379.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 375.45 | 377.44 | 379.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:15:00 | 375.95 | 377.07 | 378.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 379.95 | 377.65 | 378.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 379.50 | 377.65 | 378.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 379.35 | 377.99 | 378.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 379.35 | 377.99 | 378.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 377.65 | 377.92 | 378.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 375.80 | 377.79 | 378.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 377.15 | 377.56 | 378.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:45:00 | 376.75 | 377.22 | 377.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:30:00 | 376.20 | 377.21 | 377.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 376.00 | 376.97 | 377.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:00:00 | 375.50 | 376.55 | 377.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 375.35 | 376.31 | 377.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:30:00 | 374.75 | 375.77 | 376.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 374.80 | 375.23 | 375.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 377.45 | 375.68 | 375.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:00:00 | 377.45 | 375.68 | 375.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 377.80 | 376.10 | 375.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 377.80 | 376.10 | 375.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 380.15 | 377.04 | 376.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 396.95 | 397.61 | 393.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:30:00 | 395.55 | 397.61 | 393.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 394.35 | 397.05 | 394.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 395.00 | 397.05 | 394.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 396.75 | 396.99 | 395.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:45:00 | 398.05 | 397.22 | 395.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 10:00:00 | 397.75 | 398.70 | 397.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:30:00 | 397.70 | 398.73 | 397.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 395.70 | 397.05 | 397.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 395.70 | 397.05 | 397.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 394.30 | 396.14 | 396.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 397.65 | 395.06 | 395.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 397.65 | 395.06 | 395.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 397.65 | 395.06 | 395.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 397.65 | 395.06 | 395.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 397.60 | 395.56 | 396.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 398.25 | 395.56 | 396.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 397.05 | 396.08 | 396.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 397.00 | 396.08 | 396.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 395.80 | 396.02 | 396.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 397.40 | 396.02 | 396.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 402.85 | 397.33 | 396.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 406.70 | 400.47 | 398.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 414.20 | 414.27 | 409.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:15:00 | 417.70 | 414.30 | 410.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 13:00:00 | 417.20 | 415.49 | 411.39 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 416.05 | 416.78 | 413.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 416.40 | 416.78 | 413.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 411.70 | 415.77 | 413.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 411.70 | 415.77 | 413.26 | SL hit (close<ema400) qty=1.00 sl=413.26 alert=retest1 |

### Cycle 153 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 405.20 | 410.87 | 411.52 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 412.85 | 411.46 | 411.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 418.50 | 413.18 | 412.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 11:15:00 | 423.90 | 426.24 | 423.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 423.90 | 426.24 | 423.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 423.90 | 426.24 | 423.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 423.90 | 426.24 | 423.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 422.95 | 425.58 | 423.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 422.95 | 425.58 | 423.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 422.00 | 424.87 | 423.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 422.00 | 424.87 | 423.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 422.65 | 424.42 | 423.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 424.30 | 423.81 | 423.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 424.15 | 423.56 | 422.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:00:00 | 424.55 | 423.76 | 423.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 423.60 | 423.56 | 423.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 422.70 | 423.39 | 423.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 422.70 | 423.39 | 423.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 422.30 | 423.17 | 423.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 424.30 | 423.17 | 423.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 425.90 | 423.72 | 423.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 420.95 | 423.00 | 423.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 420.95 | 423.00 | 423.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 418.60 | 421.76 | 422.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 420.50 | 418.01 | 419.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 420.50 | 418.01 | 419.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 420.50 | 418.01 | 419.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 420.95 | 418.01 | 419.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 419.65 | 418.33 | 419.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 416.80 | 418.07 | 419.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 418.05 | 417.89 | 418.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 420.25 | 419.13 | 419.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 420.25 | 419.13 | 419.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 422.05 | 419.71 | 419.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 422.55 | 423.15 | 421.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 422.55 | 423.15 | 421.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 422.55 | 423.15 | 421.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 422.55 | 423.15 | 421.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 421.05 | 422.73 | 421.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 421.05 | 422.73 | 421.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 421.00 | 422.39 | 421.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:45:00 | 421.00 | 422.39 | 421.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 419.15 | 421.74 | 421.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 418.85 | 421.74 | 421.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 414.35 | 419.68 | 420.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 412.35 | 416.93 | 418.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 396.75 | 395.72 | 399.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 396.75 | 395.72 | 399.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 396.75 | 395.72 | 399.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 398.85 | 395.72 | 399.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 389.95 | 389.70 | 391.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 390.85 | 389.70 | 391.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 390.25 | 389.69 | 391.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 390.25 | 389.69 | 391.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 391.00 | 389.95 | 391.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 393.40 | 389.95 | 391.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 392.40 | 390.44 | 391.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 393.20 | 390.44 | 391.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 394.35 | 391.22 | 391.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 395.40 | 391.22 | 391.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 393.90 | 392.17 | 392.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 394.65 | 392.67 | 392.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 393.45 | 393.57 | 392.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 393.45 | 393.57 | 392.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 393.45 | 393.57 | 392.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:00:00 | 397.00 | 394.55 | 393.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:45:00 | 397.00 | 394.90 | 393.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 397.35 | 394.90 | 393.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:00:00 | 398.05 | 396.42 | 394.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 399.80 | 402.11 | 400.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 399.80 | 402.11 | 400.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 400.65 | 401.82 | 400.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 401.40 | 401.68 | 400.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:45:00 | 401.80 | 401.38 | 400.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 398.60 | 400.82 | 400.37 | SL hit (close<static) qty=1.00 sl=399.10 alert=retest2 |

### Cycle 159 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 397.00 | 400.06 | 400.07 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 400.40 | 400.09 | 400.07 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 399.80 | 400.04 | 400.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 397.05 | 399.44 | 399.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 10:15:00 | 399.30 | 399.11 | 399.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 10:15:00 | 399.30 | 399.11 | 399.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 399.30 | 399.11 | 399.51 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 402.25 | 400.18 | 399.96 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 398.15 | 399.83 | 399.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 395.80 | 399.03 | 399.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 398.35 | 398.18 | 398.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 14:00:00 | 398.35 | 398.18 | 398.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 398.40 | 398.22 | 398.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 398.75 | 398.22 | 398.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 398.10 | 398.20 | 398.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 397.80 | 398.20 | 398.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 395.40 | 397.64 | 398.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:00:00 | 393.95 | 396.90 | 398.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 14:45:00 | 394.10 | 394.12 | 396.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:00:00 | 394.95 | 392.68 | 394.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 394.75 | 393.48 | 394.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 396.45 | 394.07 | 394.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 396.25 | 394.07 | 394.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 396.85 | 394.63 | 394.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 396.85 | 394.63 | 394.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 398.25 | 395.35 | 394.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 395.10 | 395.64 | 395.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 395.10 | 395.64 | 395.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 395.10 | 395.64 | 395.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 394.85 | 395.64 | 395.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 397.45 | 396.01 | 395.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 394.35 | 396.01 | 395.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 395.40 | 395.88 | 395.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 395.40 | 395.88 | 395.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 399.85 | 396.68 | 395.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 394.95 | 396.68 | 395.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 398.75 | 398.97 | 397.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 398.10 | 398.97 | 397.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 398.60 | 399.14 | 398.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 398.60 | 399.14 | 398.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 397.45 | 398.80 | 398.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 396.90 | 398.80 | 398.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 396.90 | 398.42 | 398.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 396.40 | 398.42 | 398.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 393.10 | 396.96 | 397.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 392.25 | 396.02 | 396.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 11:15:00 | 393.20 | 393.00 | 394.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 12:00:00 | 393.20 | 393.00 | 394.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 384.70 | 382.23 | 384.62 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 383.85 | 383.38 | 383.37 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 14:15:00 | 382.95 | 383.30 | 383.33 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 384.00 | 383.44 | 383.39 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 381.10 | 382.97 | 383.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 380.00 | 382.03 | 382.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 380.60 | 379.42 | 380.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 380.60 | 379.42 | 380.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 380.60 | 379.42 | 380.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 380.60 | 379.42 | 380.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 380.95 | 379.73 | 380.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 379.95 | 379.77 | 380.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 380.05 | 378.60 | 378.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:00:00 | 380.25 | 378.35 | 378.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 382.00 | 379.08 | 378.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 382.00 | 379.08 | 378.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 385.65 | 381.94 | 380.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 382.40 | 382.44 | 381.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 09:15:00 | 380.50 | 382.44 | 381.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 381.00 | 382.15 | 381.39 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 380.55 | 381.13 | 381.18 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 381.95 | 381.29 | 381.25 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 376.70 | 380.54 | 380.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 375.35 | 377.75 | 379.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 365.65 | 364.20 | 367.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 365.65 | 364.20 | 367.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 364.85 | 365.06 | 366.60 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 373.00 | 367.80 | 367.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 11:15:00 | 374.50 | 370.13 | 368.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 15:15:00 | 378.30 | 378.45 | 376.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 376.75 | 378.45 | 376.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 374.40 | 377.64 | 376.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 374.40 | 377.64 | 376.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 371.75 | 376.46 | 375.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 371.75 | 376.46 | 375.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 373.95 | 375.29 | 375.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 371.55 | 374.54 | 375.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 374.65 | 374.56 | 375.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:15:00 | 376.80 | 374.56 | 375.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 377.25 | 375.10 | 375.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 378.55 | 375.10 | 375.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 377.05 | 375.49 | 375.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 11:15:00 | 378.80 | 376.15 | 375.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 378.15 | 378.59 | 377.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 378.15 | 378.59 | 377.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 378.15 | 378.59 | 377.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 377.35 | 378.59 | 377.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 378.80 | 378.63 | 377.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 377.35 | 378.63 | 377.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 378.10 | 378.53 | 377.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 378.20 | 378.53 | 377.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 377.40 | 378.86 | 378.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 377.40 | 378.86 | 378.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 375.65 | 378.21 | 377.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 375.65 | 378.21 | 377.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 373.25 | 377.22 | 377.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 372.00 | 375.54 | 376.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 374.00 | 373.61 | 375.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 374.00 | 373.61 | 375.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 373.35 | 373.72 | 374.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 374.15 | 373.72 | 374.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 373.15 | 372.14 | 373.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 373.25 | 372.14 | 373.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 373.95 | 372.50 | 373.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 373.95 | 372.50 | 373.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 373.20 | 372.64 | 373.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:30:00 | 374.20 | 372.64 | 373.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 373.15 | 372.74 | 373.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:30:00 | 373.65 | 372.74 | 373.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 374.90 | 373.17 | 373.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 374.90 | 373.17 | 373.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 374.60 | 373.46 | 373.45 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 371.75 | 373.27 | 373.39 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 374.95 | 373.62 | 373.53 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 370.70 | 373.06 | 373.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 366.85 | 371.16 | 372.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 361.50 | 360.36 | 362.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 361.50 | 360.36 | 362.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 361.50 | 360.36 | 362.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 361.40 | 360.36 | 362.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 360.60 | 360.40 | 362.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 360.60 | 360.40 | 362.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 363.20 | 361.12 | 362.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 363.20 | 361.12 | 362.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 362.90 | 361.47 | 362.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 367.25 | 361.47 | 362.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 366.55 | 363.31 | 363.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 367.95 | 364.24 | 363.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 365.80 | 365.85 | 364.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 365.70 | 365.85 | 364.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 366.80 | 366.04 | 364.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 364.55 | 366.04 | 364.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 366.95 | 366.22 | 365.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 367.70 | 366.22 | 365.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 367.50 | 369.05 | 368.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 369.25 | 368.52 | 367.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 367.90 | 368.19 | 367.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 367.60 | 368.07 | 367.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 365.90 | 367.26 | 367.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 365.90 | 367.26 | 367.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 364.95 | 366.80 | 367.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 360.50 | 359.68 | 361.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:00:00 | 360.50 | 359.68 | 361.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 362.00 | 360.36 | 361.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 362.00 | 360.36 | 361.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 362.15 | 360.72 | 361.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 362.15 | 360.72 | 361.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 362.30 | 361.24 | 361.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 363.35 | 361.24 | 361.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 362.65 | 361.63 | 361.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 362.65 | 361.63 | 361.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 362.50 | 361.80 | 361.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 362.50 | 361.80 | 361.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 362.45 | 361.93 | 361.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 363.45 | 362.24 | 362.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 367.25 | 369.60 | 367.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 367.25 | 369.60 | 367.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 367.25 | 369.60 | 367.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 367.25 | 369.60 | 367.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 367.50 | 369.18 | 367.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 366.00 | 369.18 | 367.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 366.80 | 368.70 | 367.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 366.70 | 368.70 | 367.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 366.25 | 368.21 | 367.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 365.60 | 368.21 | 367.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 363.10 | 367.19 | 366.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 363.10 | 367.19 | 366.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 364.75 | 366.26 | 366.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 362.95 | 365.60 | 366.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 346.90 | 346.30 | 350.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:30:00 | 346.90 | 346.30 | 350.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 349.30 | 346.90 | 350.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 349.30 | 346.90 | 350.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 350.25 | 347.57 | 350.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 348.40 | 347.57 | 350.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 346.05 | 347.26 | 349.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 344.50 | 346.45 | 348.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 344.55 | 345.25 | 347.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 345.00 | 345.20 | 347.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:45:00 | 345.20 | 345.03 | 346.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 345.35 | 344.96 | 346.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 346.30 | 344.96 | 346.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 348.00 | 345.57 | 346.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 348.75 | 345.57 | 346.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 347.95 | 346.04 | 346.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 348.50 | 346.04 | 346.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 327.27 | 334.04 | 338.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 327.32 | 334.04 | 338.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 327.75 | 334.04 | 338.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 327.94 | 334.04 | 338.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 331.70 | 328.70 | 332.77 | SL hit (close>ema200) qty=0.50 sl=328.70 alert=retest2 |

### Cycle 186 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 323.55 | 322.07 | 322.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 325.00 | 322.66 | 322.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 320.25 | 322.18 | 322.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 320.25 | 322.18 | 322.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 320.25 | 322.18 | 322.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 320.25 | 322.18 | 322.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 320.80 | 321.90 | 322.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 310.00 | 316.55 | 319.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 316.40 | 315.48 | 317.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:30:00 | 316.70 | 315.48 | 317.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 320.95 | 316.57 | 318.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 320.95 | 316.57 | 318.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 322.00 | 317.66 | 318.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 327.40 | 317.66 | 318.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 325.15 | 319.16 | 319.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 343.70 | 329.11 | 324.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 337.00 | 337.96 | 332.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:45:00 | 336.70 | 337.96 | 332.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 331.65 | 335.95 | 334.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 331.65 | 335.95 | 334.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 332.00 | 335.16 | 333.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 330.65 | 335.16 | 333.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 330.15 | 333.65 | 333.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 330.15 | 333.65 | 333.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 331.20 | 333.16 | 333.17 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 335.10 | 333.42 | 333.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 337.60 | 334.26 | 333.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 12:15:00 | 338.35 | 339.02 | 337.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:00:00 | 338.35 | 339.02 | 337.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 337.65 | 338.75 | 337.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 337.65 | 338.75 | 337.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 338.85 | 338.84 | 337.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 338.35 | 338.84 | 337.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 338.70 | 339.86 | 338.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 338.10 | 339.86 | 338.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 338.95 | 339.67 | 338.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 338.60 | 339.67 | 338.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 336.00 | 338.94 | 338.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 336.00 | 338.94 | 338.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 337.15 | 338.58 | 338.48 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 337.65 | 338.40 | 338.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 336.65 | 338.05 | 338.25 | Break + close below crossover candle low |

### Cycle 192 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 340.00 | 338.44 | 338.40 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 333.05 | 337.36 | 337.92 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 339.00 | 337.90 | 337.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 340.15 | 338.35 | 338.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 340.50 | 340.75 | 339.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 342.10 | 340.75 | 339.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 339.10 | 340.42 | 339.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:15:00 | 338.35 | 340.42 | 339.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 339.70 | 340.28 | 339.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 12:00:00 | 340.15 | 340.25 | 339.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 340.60 | 340.57 | 340.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 337.90 | 339.94 | 339.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 337.90 | 339.94 | 339.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 335.90 | 338.78 | 339.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 336.45 | 336.17 | 337.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 336.45 | 336.17 | 337.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 337.10 | 336.36 | 337.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 337.40 | 336.36 | 337.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 338.50 | 336.79 | 337.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 338.70 | 336.79 | 337.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 336.80 | 336.79 | 337.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 334.85 | 336.79 | 337.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 335.60 | 336.75 | 337.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 338.40 | 337.51 | 337.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 338.40 | 337.51 | 337.51 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 335.20 | 337.05 | 337.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 334.90 | 336.43 | 336.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 336.50 | 335.96 | 336.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 336.50 | 335.96 | 336.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 337.00 | 336.17 | 336.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 338.25 | 336.17 | 336.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 339.60 | 336.85 | 336.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 339.60 | 336.85 | 336.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 341.10 | 337.70 | 337.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 341.95 | 338.55 | 337.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 338.45 | 338.53 | 337.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 338.45 | 338.53 | 337.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 336.55 | 338.14 | 337.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 336.55 | 338.14 | 337.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 337.30 | 337.97 | 337.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:45:00 | 336.75 | 337.97 | 337.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 337.45 | 337.86 | 337.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 337.55 | 337.59 | 337.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 337.35 | 337.54 | 337.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 336.50 | 337.54 | 337.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 336.50 | 337.33 | 337.40 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 339.80 | 337.65 | 337.51 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 336.30 | 337.27 | 337.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 334.00 | 336.36 | 336.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 317.05 | 316.16 | 320.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 317.05 | 316.16 | 320.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 319.05 | 316.49 | 319.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 320.10 | 316.49 | 319.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 320.15 | 317.22 | 319.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 322.50 | 317.22 | 319.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 322.20 | 318.22 | 319.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 321.60 | 318.22 | 319.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 317.70 | 318.11 | 319.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 316.70 | 317.81 | 319.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 317.50 | 314.55 | 314.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 317.50 | 314.55 | 314.21 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 312.80 | 313.95 | 314.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 310.85 | 313.33 | 313.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 312.20 | 311.53 | 312.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 312.20 | 311.53 | 312.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 312.20 | 311.53 | 312.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 312.20 | 311.53 | 312.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 311.40 | 311.50 | 312.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:15:00 | 312.80 | 311.50 | 312.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 312.25 | 311.65 | 312.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 313.70 | 311.65 | 312.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 310.35 | 311.39 | 312.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 308.45 | 311.00 | 312.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 293.03 | 298.30 | 303.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 301.85 | 296.66 | 300.06 | SL hit (close>ema200) qty=0.50 sl=296.66 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 306.25 | 301.75 | 301.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 308.70 | 303.14 | 302.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 302.20 | 305.98 | 304.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 302.20 | 305.98 | 304.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 302.20 | 305.98 | 304.38 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 301.75 | 303.60 | 303.69 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 305.90 | 303.92 | 303.81 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 294.90 | 302.39 | 303.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 293.45 | 299.40 | 301.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 298.80 | 294.94 | 297.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 298.80 | 294.94 | 297.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 298.80 | 294.94 | 297.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 298.20 | 294.94 | 297.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 299.35 | 295.83 | 297.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 300.55 | 295.83 | 297.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 298.80 | 296.87 | 297.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 306.70 | 296.87 | 297.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 309.60 | 299.42 | 298.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 311.45 | 301.82 | 299.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 302.50 | 305.21 | 302.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 302.50 | 305.21 | 302.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 302.50 | 305.21 | 302.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 302.50 | 305.21 | 302.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 301.20 | 304.41 | 302.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 300.50 | 304.41 | 302.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 302.25 | 303.97 | 302.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 301.10 | 303.97 | 302.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 303.15 | 303.62 | 302.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 303.15 | 303.62 | 302.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 301.00 | 303.10 | 302.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 301.00 | 303.10 | 302.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 300.00 | 302.48 | 302.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 295.35 | 302.48 | 302.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 295.20 | 301.02 | 301.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 292.05 | 299.23 | 300.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 299.45 | 294.17 | 297.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 299.45 | 294.17 | 297.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 299.45 | 294.17 | 297.02 | EMA400 retest candle locked (from downside) |

### Cycle 210 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 299.45 | 298.40 | 298.32 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 291.15 | 297.35 | 297.88 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 298.05 | 297.28 | 297.24 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 294.70 | 297.21 | 297.26 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 314.20 | 300.09 | 298.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 314.70 | 303.01 | 299.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 307.25 | 310.19 | 305.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 307.25 | 310.19 | 305.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 307.25 | 310.19 | 305.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 307.25 | 310.19 | 305.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 324.55 | 321.76 | 316.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 328.40 | 322.95 | 317.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 328.20 | 324.00 | 318.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 326.85 | 324.80 | 319.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 328.85 | 324.82 | 320.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 330.45 | 330.36 | 328.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:30:00 | 329.95 | 330.36 | 328.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 328.55 | 330.01 | 328.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 332.15 | 330.38 | 329.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 14:15:00 | 331.50 | 331.32 | 330.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 14:45:00 | 331.00 | 331.14 | 330.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 333.00 | 331.49 | 330.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 336.45 | 333.69 | 332.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:45:00 | 340.15 | 335.29 | 333.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 12:15:00 | 359.54 | 342.95 | 337.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 341.25 | 344.14 | 344.17 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 346.85 | 344.22 | 344.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 351.00 | 346.11 | 345.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 358.65 | 362.03 | 358.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:30:00 | 360.65 | 362.03 | 358.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 357.15 | 361.05 | 358.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 357.15 | 361.05 | 358.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 357.00 | 360.24 | 357.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 357.00 | 360.24 | 357.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 356.75 | 359.54 | 357.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 356.30 | 359.54 | 357.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 359.55 | 359.54 | 357.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:30:00 | 361.60 | 359.72 | 358.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 366.90 | 359.67 | 358.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 14:00:00 | 362.80 | 366.22 | 362.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 361.00 | 364.53 | 362.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 361.00 | 363.83 | 362.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 366.15 | 363.83 | 362.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 361.95 | 363.52 | 362.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 363.00 | 362.96 | 362.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 363.80 | 362.50 | 362.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 362.75 | 362.55 | 362.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 345.65 | 358.98 | 360.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 345.65 | 358.98 | 360.73 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 363.90 | 358.91 | 358.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 365.05 | 360.14 | 359.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 361.75 | 361.79 | 360.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 15:15:00 | 361.75 | 361.79 | 360.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 361.75 | 361.79 | 360.81 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-19 13:45:00 | 203.60 | 2023-05-22 12:15:00 | 201.55 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-05-25 15:15:00 | 208.65 | 2023-05-30 14:15:00 | 207.50 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2023-05-26 12:15:00 | 208.85 | 2023-05-30 14:15:00 | 207.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-05-29 10:00:00 | 208.90 | 2023-05-30 14:15:00 | 207.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-05-29 10:30:00 | 208.95 | 2023-05-30 14:15:00 | 207.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2023-06-02 09:15:00 | 212.40 | 2023-06-02 11:15:00 | 210.25 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-06-02 13:00:00 | 212.35 | 2023-06-06 09:15:00 | 210.40 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-06-02 14:00:00 | 212.20 | 2023-06-06 09:15:00 | 210.40 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2023-06-05 09:15:00 | 212.40 | 2023-06-06 09:15:00 | 210.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-06-05 10:00:00 | 212.60 | 2023-06-06 09:15:00 | 210.40 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-06-06 09:15:00 | 211.70 | 2023-06-06 09:15:00 | 210.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-06-20 14:15:00 | 221.00 | 2023-07-05 10:15:00 | 243.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-20 15:15:00 | 220.80 | 2023-07-05 10:15:00 | 242.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2023-07-18 11:15:00 | 248.00 | 2023-07-20 09:15:00 | 249.40 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2023-07-18 13:45:00 | 247.85 | 2023-07-20 09:15:00 | 249.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-07-24 15:15:00 | 251.90 | 2023-07-25 12:15:00 | 249.95 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-07-25 09:30:00 | 251.80 | 2023-07-25 12:15:00 | 249.95 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-08-08 13:30:00 | 260.35 | 2023-08-16 11:15:00 | 263.00 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2023-08-09 10:15:00 | 261.15 | 2023-08-16 11:15:00 | 263.00 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2023-08-28 12:15:00 | 259.85 | 2023-08-31 09:15:00 | 266.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2023-08-29 14:45:00 | 259.90 | 2023-08-31 09:15:00 | 266.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2023-09-06 14:45:00 | 271.65 | 2023-09-12 09:15:00 | 269.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2023-09-14 12:15:00 | 267.30 | 2023-09-15 15:15:00 | 269.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-09-15 10:15:00 | 267.75 | 2023-09-15 15:15:00 | 269.70 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-09-18 12:15:00 | 266.75 | 2023-09-27 14:15:00 | 260.30 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2023-10-03 13:30:00 | 260.00 | 2023-10-04 09:15:00 | 258.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-10-03 14:15:00 | 260.05 | 2023-10-04 09:15:00 | 258.70 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-10-05 10:15:00 | 257.65 | 2023-10-06 12:15:00 | 259.20 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2023-10-06 09:30:00 | 257.85 | 2023-10-06 12:15:00 | 259.20 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-10-06 11:30:00 | 257.85 | 2023-10-06 12:15:00 | 259.20 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-10-10 12:15:00 | 257.75 | 2023-10-10 14:15:00 | 258.40 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-10-16 10:45:00 | 264.10 | 2023-10-19 10:15:00 | 264.70 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2023-10-16 11:30:00 | 264.45 | 2023-10-19 10:15:00 | 264.70 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2023-10-16 12:00:00 | 264.15 | 2023-10-19 10:15:00 | 264.70 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2023-10-16 13:00:00 | 264.10 | 2023-10-19 10:15:00 | 264.70 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2023-10-20 09:15:00 | 264.40 | 2023-10-23 15:15:00 | 251.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:15:00 | 264.40 | 2023-10-27 09:15:00 | 249.80 | STOP_HIT | 0.50 | 5.52% |
| BUY | retest2 | 2023-11-01 12:15:00 | 254.70 | 2023-11-01 12:15:00 | 253.25 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-11-02 09:30:00 | 254.45 | 2023-11-17 14:15:00 | 279.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-02 10:30:00 | 255.90 | 2023-11-17 14:15:00 | 281.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-29 14:15:00 | 284.95 | 2023-11-30 09:15:00 | 282.30 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-11-30 11:15:00 | 284.40 | 2023-12-08 12:15:00 | 289.50 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2023-11-30 12:15:00 | 284.70 | 2023-12-08 12:15:00 | 289.50 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-01-11 10:45:00 | 331.10 | 2024-01-15 13:15:00 | 324.85 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-01-25 10:30:00 | 311.15 | 2024-01-30 09:15:00 | 317.15 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-02-07 14:00:00 | 347.50 | 2024-02-09 09:15:00 | 341.15 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-02-08 12:00:00 | 347.60 | 2024-02-09 09:15:00 | 341.15 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-02-19 09:15:00 | 340.80 | 2024-02-20 09:15:00 | 330.90 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-02-19 15:15:00 | 337.25 | 2024-02-20 09:15:00 | 330.90 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-02-28 10:15:00 | 324.45 | 2024-03-01 14:15:00 | 323.70 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-02-28 10:45:00 | 321.85 | 2024-03-01 14:15:00 | 323.70 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-03-01 09:30:00 | 324.25 | 2024-03-01 14:15:00 | 323.70 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-03-01 14:15:00 | 323.95 | 2024-03-01 14:15:00 | 323.70 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-03-19 09:15:00 | 301.60 | 2024-03-21 10:15:00 | 306.15 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-04-22 11:45:00 | 468.70 | 2024-04-24 11:15:00 | 445.45 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2024-04-23 11:30:00 | 464.60 | 2024-04-24 11:15:00 | 445.45 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2024-04-23 13:30:00 | 464.75 | 2024-04-24 11:15:00 | 445.45 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-04-23 14:00:00 | 466.20 | 2024-04-24 11:15:00 | 445.45 | STOP_HIT | 1.00 | -4.45% |
| BUY | retest2 | 2024-04-30 09:15:00 | 460.85 | 2024-05-06 11:15:00 | 459.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-05-06 10:15:00 | 459.45 | 2024-05-06 11:15:00 | 459.65 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-05-14 09:30:00 | 452.95 | 2024-05-14 10:15:00 | 458.60 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-05-16 10:30:00 | 470.80 | 2024-05-22 12:15:00 | 467.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-05-22 09:45:00 | 471.65 | 2024-05-22 12:15:00 | 467.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-05-31 09:15:00 | 508.40 | 2024-05-31 11:15:00 | 498.20 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-06-24 11:45:00 | 569.70 | 2024-06-26 14:15:00 | 568.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-06-24 13:00:00 | 569.90 | 2024-06-26 14:15:00 | 568.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-06-24 15:00:00 | 574.05 | 2024-06-26 14:15:00 | 568.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-06-28 15:00:00 | 564.10 | 2024-07-02 10:15:00 | 571.60 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-07-01 10:00:00 | 564.80 | 2024-07-02 10:15:00 | 571.60 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-08-08 09:15:00 | 492.95 | 2024-08-09 11:15:00 | 499.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-08-21 09:15:00 | 509.05 | 2024-08-23 15:15:00 | 501.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-08-27 14:45:00 | 498.05 | 2024-09-05 13:15:00 | 498.05 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-28 09:15:00 | 498.65 | 2024-09-05 13:15:00 | 498.05 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-09-11 13:30:00 | 476.90 | 2024-09-13 10:15:00 | 481.55 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-12 11:30:00 | 476.70 | 2024-09-13 10:15:00 | 481.55 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-10-08 14:30:00 | 491.30 | 2024-10-08 15:15:00 | 494.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-10-10 15:15:00 | 513.00 | 2024-10-16 10:15:00 | 516.50 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2024-11-04 09:15:00 | 449.95 | 2024-11-06 14:15:00 | 451.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-11-27 09:15:00 | 430.45 | 2024-12-12 13:15:00 | 462.50 | STOP_HIT | 1.00 | 7.45% |
| SELL | retest2 | 2024-12-20 11:30:00 | 440.40 | 2024-12-20 15:15:00 | 418.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:15:00 | 439.70 | 2024-12-20 15:15:00 | 417.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 11:30:00 | 440.40 | 2024-12-26 12:15:00 | 417.90 | STOP_HIT | 0.50 | 5.11% |
| SELL | retest2 | 2024-12-20 12:15:00 | 439.70 | 2024-12-26 12:15:00 | 417.90 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2025-01-20 12:30:00 | 390.50 | 2025-01-21 09:15:00 | 386.15 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-29 12:00:00 | 352.90 | 2025-01-30 11:15:00 | 360.25 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-01-29 15:15:00 | 352.10 | 2025-01-30 11:15:00 | 360.25 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-02-04 09:15:00 | 380.40 | 2025-02-06 15:15:00 | 378.95 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-02-13 14:00:00 | 366.75 | 2025-02-17 09:15:00 | 348.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:30:00 | 366.75 | 2025-02-17 09:15:00 | 348.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:00:00 | 366.75 | 2025-02-17 11:15:00 | 358.00 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2025-02-13 14:30:00 | 366.75 | 2025-02-17 11:15:00 | 358.00 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2025-02-14 09:15:00 | 364.30 | 2025-02-19 10:15:00 | 364.40 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-02-19 10:00:00 | 365.25 | 2025-02-19 10:15:00 | 364.40 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-03-04 12:45:00 | 344.70 | 2025-03-05 09:15:00 | 355.40 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-03-04 15:00:00 | 345.55 | 2025-03-05 09:15:00 | 355.40 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-03-07 14:30:00 | 359.10 | 2025-03-10 11:15:00 | 351.90 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-03-26 09:15:00 | 373.30 | 2025-03-26 15:15:00 | 357.50 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2025-03-27 09:45:00 | 361.00 | 2025-03-27 10:15:00 | 359.40 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-03-28 12:15:00 | 358.75 | 2025-04-01 09:15:00 | 364.75 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-04-08 10:30:00 | 355.15 | 2025-04-09 13:15:00 | 362.20 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-04-09 09:15:00 | 354.15 | 2025-04-09 13:15:00 | 362.20 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-04-17 11:15:00 | 377.00 | 2025-04-25 09:15:00 | 373.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-04-21 09:30:00 | 379.05 | 2025-04-25 09:15:00 | 373.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-04-30 13:45:00 | 362.80 | 2025-05-06 12:15:00 | 372.70 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-05-23 13:30:00 | 382.80 | 2025-05-26 14:15:00 | 387.05 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-05-23 14:30:00 | 382.65 | 2025-05-26 14:15:00 | 387.05 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-05-29 11:15:00 | 388.05 | 2025-06-03 12:15:00 | 385.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-05-29 11:45:00 | 388.90 | 2025-06-03 12:15:00 | 385.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-06-02 09:15:00 | 387.90 | 2025-06-03 13:15:00 | 386.25 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-02 13:15:00 | 389.95 | 2025-06-03 13:15:00 | 386.25 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-03 09:15:00 | 390.65 | 2025-06-03 13:15:00 | 386.25 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-03 12:00:00 | 387.75 | 2025-06-03 13:15:00 | 386.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-06-23 09:15:00 | 379.40 | 2025-06-24 09:15:00 | 386.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-06-23 11:45:00 | 378.60 | 2025-06-24 09:15:00 | 386.80 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-06-23 13:45:00 | 379.70 | 2025-06-24 09:15:00 | 386.80 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-06-23 14:15:00 | 379.15 | 2025-06-24 09:15:00 | 386.80 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2025-06-26 09:15:00 | 387.05 | 2025-06-30 09:15:00 | 386.70 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-06-26 12:15:00 | 387.10 | 2025-07-01 11:15:00 | 385.75 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-06-26 14:15:00 | 387.30 | 2025-07-01 11:15:00 | 385.75 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-06-30 10:45:00 | 388.65 | 2025-07-01 11:15:00 | 385.75 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-01 10:30:00 | 387.10 | 2025-07-01 11:15:00 | 385.75 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-07-08 12:45:00 | 386.00 | 2025-07-11 11:15:00 | 383.95 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-08-04 11:30:00 | 381.55 | 2025-08-04 12:15:00 | 387.35 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-06 09:15:00 | 386.85 | 2025-08-06 09:15:00 | 382.10 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-08 10:45:00 | 375.80 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-08-08 12:00:00 | 376.00 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-08-08 15:00:00 | 375.45 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-08-11 11:15:00 | 375.95 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-08-12 09:30:00 | 375.80 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-08-12 12:30:00 | 377.15 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-08-12 14:45:00 | 376.75 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-08-13 10:30:00 | 376.20 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-13 14:00:00 | 375.50 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-13 15:00:00 | 375.35 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-08-14 09:30:00 | 374.75 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-18 12:00:00 | 374.80 | 2025-08-18 13:15:00 | 377.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-08-25 10:45:00 | 398.05 | 2025-08-28 11:15:00 | 395.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-08-26 10:00:00 | 397.75 | 2025-08-28 11:15:00 | 395.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-08-26 12:30:00 | 397.70 | 2025-08-28 11:15:00 | 395.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-09-03 11:15:00 | 417.70 | 2025-09-04 10:15:00 | 411.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-09-03 13:00:00 | 417.20 | 2025-09-04 10:15:00 | 411.70 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-09-11 09:30:00 | 424.30 | 2025-09-12 12:15:00 | 420.95 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-09-11 11:15:00 | 424.15 | 2025-09-12 12:15:00 | 420.95 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-11 12:00:00 | 424.55 | 2025-09-12 12:15:00 | 420.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-11 13:30:00 | 423.60 | 2025-09-12 12:15:00 | 420.95 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-16 11:45:00 | 416.80 | 2025-09-18 09:15:00 | 420.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-17 09:45:00 | 418.05 | 2025-09-18 09:15:00 | 420.25 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-03 13:00:00 | 397.00 | 2025-10-08 14:15:00 | 398.60 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-10-03 13:45:00 | 397.00 | 2025-10-08 14:15:00 | 398.60 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-10-03 14:15:00 | 397.35 | 2025-10-08 15:15:00 | 397.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-06 10:00:00 | 398.05 | 2025-10-08 15:15:00 | 397.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-10-08 12:45:00 | 401.40 | 2025-10-08 15:15:00 | 397.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-08 13:45:00 | 401.80 | 2025-10-08 15:15:00 | 397.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-14 11:00:00 | 393.95 | 2025-10-16 13:15:00 | 396.85 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-10-14 14:45:00 | 394.10 | 2025-10-16 13:15:00 | 396.85 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-16 10:00:00 | 394.95 | 2025-10-16 13:15:00 | 396.85 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-10-16 11:30:00 | 394.75 | 2025-10-16 13:15:00 | 396.85 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-06 15:00:00 | 379.95 | 2025-11-12 11:15:00 | 382.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-10 14:30:00 | 380.05 | 2025-11-12 11:15:00 | 382.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-12 11:00:00 | 380.25 | 2025-11-12 11:15:00 | 382.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-23 11:15:00 | 367.70 | 2025-12-26 13:15:00 | 365.90 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-24 14:45:00 | 367.50 | 2025-12-26 13:15:00 | 365.90 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-12-26 09:15:00 | 369.25 | 2025-12-26 13:15:00 | 365.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-26 10:15:00 | 367.90 | 2025-12-26 13:15:00 | 365.90 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-13 12:00:00 | 344.50 | 2026-01-21 09:15:00 | 327.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 344.55 | 2026-01-21 09:15:00 | 327.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 345.00 | 2026-01-21 09:15:00 | 327.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:45:00 | 345.20 | 2026-01-21 09:15:00 | 327.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 344.50 | 2026-01-22 09:15:00 | 331.70 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2026-01-14 09:30:00 | 344.55 | 2026-01-22 09:15:00 | 331.70 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2026-01-14 11:00:00 | 345.00 | 2026-01-22 09:15:00 | 331.70 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2026-01-14 11:45:00 | 345.20 | 2026-01-22 09:15:00 | 331.70 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2026-01-23 10:45:00 | 332.75 | 2026-01-29 09:15:00 | 316.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 332.75 | 2026-01-29 09:15:00 | 316.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 332.75 | 2026-01-29 13:15:00 | 319.40 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2026-01-23 11:15:00 | 332.75 | 2026-01-29 13:15:00 | 319.40 | STOP_HIT | 0.50 | 4.01% |
| BUY | retest2 | 2026-02-18 12:00:00 | 340.15 | 2026-02-19 11:15:00 | 337.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2026-02-19 10:15:00 | 340.60 | 2026-02-19 11:15:00 | 337.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-02-20 15:15:00 | 334.85 | 2026-02-23 15:15:00 | 338.40 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-23 10:30:00 | 335.60 | 2026-02-23 15:15:00 | 338.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-06 14:45:00 | 316.70 | 2026-03-11 09:15:00 | 317.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-03-13 09:15:00 | 308.45 | 2026-03-16 10:15:00 | 293.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 308.45 | 2026-03-17 09:15:00 | 301.85 | STOP_HIT | 0.50 | 2.14% |
| BUY | retest2 | 2026-04-13 10:45:00 | 328.40 | 2026-04-22 12:15:00 | 359.54 | TARGET_HIT | 1.00 | 9.48% |
| BUY | retest2 | 2026-04-13 12:00:00 | 328.20 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 3.98% |
| BUY | retest2 | 2026-04-13 13:45:00 | 326.85 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 4.41% |
| BUY | retest2 | 2026-04-15 09:15:00 | 328.85 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 3.77% |
| BUY | retest2 | 2026-04-20 10:30:00 | 332.15 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2026-04-20 14:15:00 | 331.50 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 2.94% |
| BUY | retest2 | 2026-04-20 14:45:00 | 331.00 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2026-04-21 09:30:00 | 333.00 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2026-04-22 10:45:00 | 340.15 | 2026-04-24 13:15:00 | 341.25 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-04-30 14:30:00 | 361.60 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2026-05-04 09:15:00 | 366.90 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -5.79% |
| BUY | retest2 | 2026-05-04 14:00:00 | 362.80 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2026-05-04 15:15:00 | 361.00 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2026-05-05 09:15:00 | 366.15 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2026-05-05 12:15:00 | 361.95 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2026-05-05 14:15:00 | 363.00 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2026-05-06 09:15:00 | 363.80 | 2026-05-06 12:15:00 | 345.65 | STOP_HIT | 1.00 | -4.99% |
