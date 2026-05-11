# Shipping Corporation of India Ltd. (SCI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 339.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 17 |
| TARGET_HIT | 10 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 12
- **Target hits / Stop hits / Partials:** 10 / 20 / 17
- **Avg / median % per leg:** 3.92% / 5.00%
- **Sum % (uncompounded):** 184.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | 0.03% | 0.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | 0.03% | 0.2% |
| SELL (all) | 39 | 33 | 84.6% | 8 | 14 | 17 | 4.71% | 183.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 33 | 84.6% | 8 | 14 | 17 | 4.71% | 183.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 35 | 74.5% | 10 | 20 | 17 | 3.92% | 184.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 242.15 | 266.99 | 267.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 12:15:00 | 241.00 | 260.46 | 263.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 259.80 | 259.38 | 262.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 14:15:00 | 259.80 | 259.38 | 262.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 259.80 | 259.38 | 262.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 259.80 | 259.38 | 262.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 264.50 | 259.43 | 262.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 265.95 | 259.43 | 262.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 267.60 | 259.52 | 262.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 267.60 | 259.52 | 262.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 266.45 | 260.72 | 263.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 266.45 | 260.72 | 263.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 266.45 | 260.93 | 263.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 266.45 | 260.93 | 263.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 267.95 | 261.00 | 263.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:30:00 | 269.80 | 261.00 | 263.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 238.89 | 228.41 | 240.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:45:00 | 240.71 | 228.41 | 240.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 239.88 | 228.53 | 240.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 238.80 | 228.53 | 240.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 242.57 | 228.67 | 240.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:00:00 | 242.57 | 228.67 | 240.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 238.60 | 228.77 | 240.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 235.89 | 228.77 | 240.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 13:15:00 | 224.10 | 228.66 | 239.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 14:15:00 | 212.30 | 227.74 | 238.68 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 192.40 | 175.25 | 175.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 198.51 | 175.48 | 175.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 217.73 | 217.77 | 207.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:30:00 | 218.32 | 217.77 | 207.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 214.94 | 218.21 | 211.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 217.77 | 212.81 | 210.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 218.98 | 212.81 | 210.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 217.16 | 213.23 | 211.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 209.40 | 214.22 | 212.06 | SL hit (close<static) qty=1.00 sl=210.08 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 221.98 | 235.22 | 235.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 219.95 | 235.06 | 235.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 227.60 | 226.62 | 230.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 227.60 | 226.62 | 230.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 234.81 | 226.63 | 230.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 233.58 | 226.63 | 230.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 233.99 | 226.71 | 230.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 229.43 | 227.05 | 230.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 231.40 | 227.41 | 230.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 231.64 | 227.45 | 230.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 231.62 | 227.53 | 230.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 229.64 | 227.70 | 230.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 230.83 | 227.70 | 230.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 232.00 | 227.74 | 230.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 232.00 | 227.74 | 230.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 231.40 | 227.78 | 230.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:45:00 | 231.03 | 228.09 | 230.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 230.63 | 228.09 | 230.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 231.00 | 228.15 | 230.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.83 | 227.96 | 230.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 220.06 | 227.96 | 230.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 220.04 | 227.96 | 230.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.48 | 227.96 | 230.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.10 | 227.96 | 230.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.45 | 227.96 | 230.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 217.96 | 227.67 | 229.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 208.48 | 226.77 | 229.39 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 266.96 | 226.29 | 226.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 268.40 | 226.71 | 226.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 249.80 | 249.96 | 241.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 11:00:00 | 249.80 | 249.96 | 241.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 238.45 | 249.76 | 241.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 240.80 | 249.76 | 241.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 233.00 | 249.59 | 241.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 232.95 | 249.59 | 241.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 240.50 | 249.33 | 241.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:00:00 | 240.50 | 249.33 | 241.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 242.40 | 249.26 | 241.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:30:00 | 241.95 | 249.26 | 241.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 242.80 | 249.20 | 241.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 242.95 | 249.20 | 241.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 243.25 | 249.14 | 241.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:15:00 | 239.15 | 249.14 | 241.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 239.00 | 249.04 | 241.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 239.40 | 249.04 | 241.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 243.05 | 248.98 | 241.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 14:30:00 | 244.10 | 247.20 | 240.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:30:00 | 246.00 | 247.00 | 241.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 245.00 | 247.20 | 241.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 15:15:00 | 238.50 | 246.83 | 241.36 | SL hit (close<static) qty=1.00 sl=238.60 alert=retest2 |

### Cycle 5 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 222.91 | 237.86 | 237.89 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 247.82 | 237.79 | 237.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 249.32 | 237.91 | 237.84 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-11 13:15:00 | 235.89 | 2024-11-12 13:15:00 | 224.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:15:00 | 235.89 | 2024-11-13 14:15:00 | 212.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-28 10:15:00 | 237.64 | 2024-12-13 10:15:00 | 225.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 09:45:00 | 237.70 | 2024-12-13 10:15:00 | 225.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 13:30:00 | 237.89 | 2024-12-13 10:15:00 | 226.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-28 10:15:00 | 237.64 | 2024-12-13 14:15:00 | 230.90 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2024-12-03 09:45:00 | 237.70 | 2024-12-13 14:15:00 | 230.90 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2024-12-03 13:30:00 | 237.89 | 2024-12-13 14:15:00 | 230.90 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2025-03-20 15:00:00 | 176.64 | 2025-03-26 14:15:00 | 167.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-20 15:00:00 | 176.64 | 2025-03-26 14:15:00 | 167.30 | STOP_HIT | 0.50 | 5.29% |
| SELL | retest2 | 2025-03-24 10:30:00 | 178.15 | 2025-03-26 14:15:00 | 169.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 10:30:00 | 178.15 | 2025-03-26 14:15:00 | 167.30 | STOP_HIT | 0.50 | 6.09% |
| SELL | retest2 | 2025-03-24 11:00:00 | 178.27 | 2025-03-26 14:15:00 | 169.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 11:00:00 | 178.27 | 2025-03-26 14:15:00 | 167.30 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2025-04-25 09:15:00 | 177.55 | 2025-04-28 09:15:00 | 168.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:15:00 | 177.55 | 2025-04-28 09:15:00 | 175.50 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2025-04-25 11:45:00 | 172.53 | 2025-04-28 09:15:00 | 175.50 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-04-28 09:15:00 | 170.68 | 2025-04-28 09:15:00 | 175.50 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-05-06 11:15:00 | 172.55 | 2025-05-08 14:15:00 | 163.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:15:00 | 172.55 | 2025-05-12 13:15:00 | 173.10 | STOP_HIT | 0.50 | -0.32% |
| SELL | retest2 | 2025-05-12 14:15:00 | 172.74 | 2025-05-13 11:15:00 | 175.68 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-08-22 10:30:00 | 217.77 | 2025-09-05 09:15:00 | 209.40 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-08-22 11:15:00 | 218.98 | 2025-09-05 09:15:00 | 209.40 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2025-09-01 14:15:00 | 217.16 | 2025-09-05 09:15:00 | 209.40 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-09-11 09:45:00 | 217.35 | 2025-09-22 09:15:00 | 234.76 | TARGET_HIT | 1.00 | 8.01% |
| BUY | retest2 | 2025-09-12 09:15:00 | 213.42 | 2025-10-23 09:15:00 | 239.09 | TARGET_HIT | 1.00 | 12.03% |
| SELL | retest2 | 2025-12-30 09:15:00 | 229.43 | 2026-01-08 12:15:00 | 219.83 | PARTIAL | 0.50 | 4.18% |
| SELL | retest2 | 2025-12-31 14:15:00 | 231.40 | 2026-01-08 12:15:00 | 220.06 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-12-31 15:15:00 | 231.64 | 2026-01-08 12:15:00 | 220.04 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-01-01 09:45:00 | 231.62 | 2026-01-08 12:15:00 | 219.48 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2026-01-05 10:45:00 | 231.03 | 2026-01-08 12:15:00 | 219.10 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-01-05 11:15:00 | 230.63 | 2026-01-08 12:15:00 | 219.45 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-05 13:00:00 | 231.00 | 2026-01-08 15:15:00 | 217.96 | PARTIAL | 0.50 | 5.65% |
| SELL | retest2 | 2025-12-30 09:15:00 | 229.43 | 2026-01-12 09:15:00 | 208.48 | TARGET_HIT | 0.50 | 9.13% |
| SELL | retest2 | 2025-12-31 14:15:00 | 231.40 | 2026-01-12 09:15:00 | 208.46 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2025-12-31 15:15:00 | 231.64 | 2026-01-13 14:15:00 | 208.26 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2026-01-01 09:45:00 | 231.62 | 2026-01-20 09:15:00 | 207.93 | TARGET_HIT | 0.50 | 10.23% |
| SELL | retest2 | 2026-01-05 10:45:00 | 231.03 | 2026-01-20 09:15:00 | 207.57 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-01-05 11:15:00 | 230.63 | 2026-01-20 09:15:00 | 207.90 | TARGET_HIT | 0.50 | 9.86% |
| SELL | retest2 | 2026-01-05 13:00:00 | 231.00 | 2026-01-20 11:15:00 | 206.49 | TARGET_HIT | 0.50 | 10.61% |
| SELL | retest2 | 2026-01-30 10:15:00 | 227.65 | 2026-02-01 09:15:00 | 232.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-01-30 11:45:00 | 226.89 | 2026-02-01 09:15:00 | 232.50 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-02-01 11:30:00 | 226.75 | 2026-02-01 12:15:00 | 215.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 11:30:00 | 226.75 | 2026-02-01 12:15:00 | 220.65 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2026-03-10 14:30:00 | 244.10 | 2026-03-13 15:15:00 | 238.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-12 10:30:00 | 246.00 | 2026-03-13 15:15:00 | 238.50 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-03-13 10:15:00 | 245.00 | 2026-03-13 15:15:00 | 238.50 | STOP_HIT | 1.00 | -2.65% |
