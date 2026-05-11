# Housing & Urban Development Corporation Ltd. (HUDCO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 232.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 33 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 28
- **Target hits / Stop hits / Partials:** 6 / 31 / 7
- **Avg / median % per leg:** -0.16% / -2.47%
- **Sum % (uncompounded):** -6.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 1 | 5.6% | 1 | 17 | 0 | -2.03% | -36.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 1 | 5.6% | 1 | 17 | 0 | -2.03% | -36.5% |
| SELL (all) | 26 | 15 | 57.7% | 5 | 14 | 7 | 1.14% | 29.7% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 4 | 2 | -1.25% | -7.5% |
| SELL @ 3rd Alert (retest2) | 20 | 11 | 55.0% | 5 | 10 | 5 | 1.86% | 37.1% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | -1.25% | -7.5% |
| retest2 (combined) | 38 | 12 | 31.6% | 6 | 27 | 5 | 0.02% | 0.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 14:15:00 | 251.50 | 283.20 | 283.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 12:15:00 | 249.25 | 279.67 | 281.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 222.70 | 222.06 | 237.42 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 14:30:00 | 219.71 | 222.65 | 236.20 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-11 11:15:00 | 219.64 | 222.55 | 235.95 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 208.72 | 221.47 | 234.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 208.66 | 221.47 | 234.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 217.61 | 215.85 | 228.97 | SL hit (close>ema200) qty=0.50 sl=215.85 alert=retest1 |

### Cycle 2 — SELL (started 2024-11-25 11:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 11:30:00 | 218.55 | 215.90 | 228.86 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 11:15:00 | 219.34 | 215.85 | 228.02 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 240.14 | 216.28 | 227.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 240.14 | 216.28 | 227.88 | SL hit (close>ema400) qty=1.00 sl=227.88 alert=retest1 |

### Cycle 3 — BUY (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 13:15:00 | 257.27 | 234.47 | 234.36 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 213.90 | 235.18 | 235.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 208.41 | 234.67 | 234.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 230.73 | 230.52 | 232.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-17 10:00:00 | 230.73 | 230.52 | 232.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 229.69 | 230.51 | 232.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 221.69 | 230.91 | 232.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 15:15:00 | 228.00 | 230.39 | 232.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 12:15:00 | 216.60 | 230.07 | 232.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 210.61 | 229.10 | 231.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 10:15:00 | 205.20 | 228.92 | 231.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 234.97 | 204.24 | 204.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 239.90 | 206.28 | 205.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 12:15:00 | 208.72 | 216.18 | 211.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 12:15:00 | 208.72 | 216.18 | 211.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 208.72 | 216.18 | 211.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 208.72 | 216.18 | 211.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 214.10 | 216.16 | 211.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:15:00 | 215.62 | 216.16 | 211.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:45:00 | 215.13 | 216.10 | 211.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 14:15:00 | 216.35 | 216.15 | 211.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 207.50 | 215.96 | 211.58 | SL hit (close<static) qty=1.00 sl=208.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 217.10 | 227.37 | 227.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 212.40 | 226.94 | 227.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 215.05 | 214.67 | 218.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 215.05 | 214.67 | 218.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 218.49 | 214.73 | 218.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:45:00 | 218.44 | 214.73 | 218.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 217.70 | 215.13 | 218.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 216.35 | 215.13 | 218.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 215.67 | 215.15 | 218.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:30:00 | 216.16 | 215.17 | 218.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 216.37 | 215.28 | 218.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 217.38 | 215.34 | 218.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 218.20 | 215.34 | 218.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 217.10 | 215.38 | 218.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:30:00 | 216.72 | 215.39 | 218.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 216.75 | 215.47 | 218.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 224.13 | 215.59 | 218.28 | SL hit (close>static) qty=1.00 sl=220.84 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 236.09 | 220.38 | 220.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 237.36 | 227.20 | 225.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 228.70 | 229.35 | 226.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 228.70 | 229.35 | 226.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 227.21 | 229.33 | 226.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 227.21 | 229.33 | 226.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 223.43 | 229.21 | 226.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 223.43 | 229.21 | 226.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 226.17 | 229.18 | 226.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 227.15 | 229.18 | 226.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:30:00 | 227.39 | 229.14 | 226.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 226.80 | 229.30 | 226.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 226.80 | 229.04 | 226.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 226.50 | 229.02 | 226.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 225.30 | 229.02 | 226.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 227.00 | 228.97 | 226.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 231.87 | 228.93 | 226.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 09:30:00 | 228.86 | 231.39 | 228.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 226.00 | 233.19 | 230.27 | SL hit (close<static) qty=1.00 sl=226.20 alert=retest2 |

### Cycle 8 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 213.19 | 227.80 | 227.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 211.27 | 225.08 | 226.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 222.25 | 220.48 | 223.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 222.25 | 220.48 | 223.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 224.05 | 220.56 | 223.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 225.05 | 220.56 | 223.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 224.10 | 220.59 | 223.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 224.74 | 220.59 | 223.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 223.99 | 220.63 | 223.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:45:00 | 224.34 | 220.63 | 223.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 224.18 | 220.97 | 223.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:00:00 | 224.18 | 220.97 | 223.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 224.81 | 221.01 | 223.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 224.45 | 221.01 | 223.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 224.85 | 221.07 | 223.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 227.30 | 221.07 | 223.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 224.98 | 222.79 | 224.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 224.50 | 222.79 | 224.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 220.90 | 223.04 | 224.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 220.18 | 223.04 | 224.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 219.74 | 223.01 | 224.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 218.42 | 222.98 | 224.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 209.17 | 220.18 | 222.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 208.75 | 220.18 | 222.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 207.50 | 219.70 | 222.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 198.16 | 216.85 | 220.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 214.51 | 191.30 | 191.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 216.87 | 191.55 | 191.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-11-08 14:30:00 | 219.71 | 2024-11-13 09:15:00 | 208.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-11 11:15:00 | 219.64 | 2024-11-13 09:15:00 | 208.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-08 14:30:00 | 219.71 | 2024-11-25 09:15:00 | 217.61 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2024-11-11 11:15:00 | 219.64 | 2024-11-25 09:15:00 | 217.61 | STOP_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2024-11-25 11:30:00 | 218.55 | 2024-11-28 09:15:00 | 240.14 | STOP_HIT | 1.00 | -9.88% |
| SELL | retest1 | 2024-11-27 11:15:00 | 219.34 | 2024-11-28 09:15:00 | 240.14 | STOP_HIT | 1.00 | -9.48% |
| SELL | retest2 | 2024-11-28 11:15:00 | 230.92 | 2024-12-02 09:15:00 | 243.91 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2024-11-28 12:00:00 | 230.87 | 2024-12-02 09:15:00 | 243.91 | STOP_HIT | 1.00 | -5.65% |
| SELL | retest2 | 2025-01-22 09:15:00 | 221.69 | 2025-01-23 12:15:00 | 216.60 | PARTIAL | 0.50 | 2.30% |
| SELL | retest2 | 2025-01-22 15:15:00 | 228.00 | 2025-01-27 09:15:00 | 210.61 | PARTIAL | 0.50 | 7.63% |
| SELL | retest2 | 2025-01-22 09:15:00 | 221.69 | 2025-01-27 10:15:00 | 205.20 | TARGET_HIT | 0.50 | 7.44% |
| SELL | retest2 | 2025-01-22 15:15:00 | 228.00 | 2025-01-31 11:15:00 | 226.80 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2025-02-01 11:45:00 | 225.00 | 2025-02-03 09:15:00 | 202.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-17 10:45:00 | 228.43 | 2025-04-21 09:15:00 | 234.97 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-05-07 14:15:00 | 215.62 | 2025-05-09 09:15:00 | 207.50 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-05-08 10:45:00 | 215.13 | 2025-05-09 09:15:00 | 207.50 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-05-08 14:15:00 | 216.35 | 2025-05-09 09:15:00 | 207.50 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2025-05-12 09:15:00 | 219.60 | 2025-05-28 09:15:00 | 241.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 10:30:00 | 228.03 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-06-13 11:15:00 | 228.00 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-06-13 15:00:00 | 228.15 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-06-16 13:30:00 | 229.41 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-06-23 10:15:00 | 229.75 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-06-23 13:00:00 | 229.68 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-07-07 09:45:00 | 229.51 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-07-08 15:15:00 | 229.35 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-09-09 10:15:00 | 216.35 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-09-09 11:45:00 | 215.67 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-09-09 13:30:00 | 216.16 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-09-10 13:00:00 | 216.37 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-09-11 12:30:00 | 216.72 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-09-12 11:30:00 | 216.75 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-11-07 11:15:00 | 227.15 | 2025-12-03 14:15:00 | 226.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-07 12:30:00 | 227.39 | 2025-12-03 14:15:00 | 226.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-11-11 13:45:00 | 226.80 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-11-14 09:30:00 | 226.80 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-11-17 09:15:00 | 231.87 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2025-11-25 09:30:00 | 228.86 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-01-08 11:15:00 | 220.18 | 2026-01-20 09:15:00 | 209.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 219.74 | 2026-01-20 09:15:00 | 208.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 12:30:00 | 218.42 | 2026-01-20 13:15:00 | 207.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 220.18 | 2026-01-23 13:15:00 | 198.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 219.74 | 2026-01-23 13:15:00 | 197.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 12:30:00 | 218.42 | 2026-01-23 13:15:00 | 196.58 | TARGET_HIT | 0.50 | 10.00% |
