# Angel One Ltd. (ANGELONE)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 326.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 4 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 23
- **Target hits / Stop hits / Partials:** 5 / 24 / 4
- **Avg / median % per leg:** 0.45% / -1.11%
- **Sum % (uncompounded):** 14.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 3 | 23.1% | 3 | 10 | 0 | 0.83% | 10.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 3 | 23.1% | 3 | 10 | 0 | 0.83% | 10.8% |
| SELL (all) | 20 | 7 | 35.0% | 2 | 14 | 4 | 0.19% | 3.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 7 | 35.0% | 2 | 14 | 4 | 0.19% | 3.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 10 | 30.3% | 5 | 24 | 4 | 0.45% | 14.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 285.03 | 311.38 | 311.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 281.00 | 308.72 | 310.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 287.90 | 280.66 | 292.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 10:00:00 | 287.90 | 280.66 | 292.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 293.24 | 280.91 | 292.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:00:00 | 293.24 | 280.91 | 292.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 294.00 | 281.04 | 292.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:45:00 | 294.38 | 281.04 | 292.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 298.14 | 281.21 | 292.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 298.14 | 281.21 | 292.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 295.14 | 288.74 | 294.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:30:00 | 295.98 | 288.74 | 294.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 295.42 | 288.81 | 294.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:00:00 | 295.42 | 288.81 | 294.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 295.10 | 288.87 | 294.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 09:30:00 | 291.70 | 289.75 | 294.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 277.11 | 289.42 | 294.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 293.70 | 288.84 | 293.46 | SL hit (close>ema200) qty=0.50 sl=288.84 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 258.65 | 241.82 | 241.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 10:15:00 | 264.62 | 245.22 | 243.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 247.85 | 249.57 | 246.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:45:00 | 248.30 | 249.57 | 246.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 245.91 | 249.53 | 246.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:45:00 | 246.01 | 249.53 | 246.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 246.29 | 249.50 | 246.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:30:00 | 247.11 | 249.41 | 246.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-01 10:15:00 | 271.82 | 250.39 | 247.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 244.62 | 287.96 | 287.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 234.36 | 287.02 | 287.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 12:15:00 | 252.91 | 251.72 | 264.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 12:45:00 | 253.21 | 251.72 | 264.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 232.67 | 218.84 | 233.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:30:00 | 232.31 | 218.84 | 233.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 233.73 | 219.25 | 233.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:45:00 | 233.51 | 219.25 | 233.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 233.80 | 219.39 | 233.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 240.40 | 219.39 | 233.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 240.10 | 219.60 | 233.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:15:00 | 241.20 | 219.60 | 233.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 233.01 | 220.99 | 233.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 233.61 | 220.99 | 233.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 232.63 | 221.11 | 233.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 232.63 | 221.11 | 233.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 233.59 | 221.35 | 233.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:45:00 | 233.92 | 221.35 | 233.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 230.53 | 221.44 | 233.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:15:00 | 229.00 | 221.44 | 233.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 227.66 | 221.60 | 233.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 228.94 | 221.60 | 233.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 236.98 | 222.97 | 233.47 | SL hit (close>static) qty=1.00 sl=234.15 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 256.15 | 235.33 | 235.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 264.20 | 237.96 | 236.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 287.36 | 290.12 | 271.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 285.92 | 290.12 | 271.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 278.95 | 290.49 | 279.06 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 258.43 | 274.41 | 274.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 256.58 | 274.23 | 274.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 272.50 | 267.17 | 270.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 269.35 | 267.19 | 270.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 268.47 | 267.19 | 270.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:45:00 | 261.39 | 267.38 | 270.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 12:15:00 | 255.05 | 267.29 | 269.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:15:00 | 248.32 | 265.79 | 269.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-26 09:15:00 | 241.62 | 264.33 | 268.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 261.70 | 243.60 | 243.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 263.34 | 244.33 | 243.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 265.38 | 265.38 | 257.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:45:00 | 266.47 | 265.38 | 257.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 258.00 | 265.05 | 258.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 258.00 | 265.05 | 258.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 255.00 | 264.95 | 258.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 254.86 | 264.95 | 258.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 256.75 | 262.29 | 257.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 257.85 | 262.25 | 257.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 258.75 | 262.20 | 257.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 257.86 | 262.11 | 257.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 257.76 | 262.06 | 257.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 257.52 | 262.01 | 257.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:45:00 | 258.60 | 261.99 | 257.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:00:00 | 257.78 | 261.90 | 257.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:45:00 | 258.19 | 261.82 | 257.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:00:00 | 257.80 | 261.78 | 257.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 257.17 | 261.73 | 257.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 257.17 | 261.73 | 257.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 258.42 | 261.70 | 257.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:30:00 | 256.40 | 261.70 | 257.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 255.33 | 261.60 | 257.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 255.33 | 261.60 | 257.51 | SL hit (close<static) qty=1.00 sl=256.05 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 10:15:00 | 234.29 | 254.90 | 254.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 09:15:00 | 233.14 | 253.72 | 254.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 249.90 | 247.20 | 250.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 249.90 | 247.20 | 250.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 249.90 | 247.20 | 250.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 251.62 | 247.20 | 250.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 250.06 | 247.23 | 250.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 250.07 | 247.23 | 250.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 248.86 | 247.25 | 250.38 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 258.27 | 252.91 | 252.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 259.76 | 253.34 | 253.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 252.00 | 253.41 | 253.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 252.00 | 253.41 | 253.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 252.00 | 253.41 | 253.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 244.00 | 253.41 | 253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 244.24 | 253.32 | 253.11 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 235.06 | 252.86 | 252.88 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 264.35 | 252.87 | 252.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 268.88 | 254.06 | 253.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 258.88 | 259.80 | 256.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 258.88 | 259.80 | 256.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 258.88 | 259.80 | 256.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 265.07 | 259.37 | 256.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 242.20 | 256.55 | 255.63 | SL hit (close<static) qty=1.00 sl=244.10 alert=retest2 |

### Cycle 11 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 235.80 | 254.76 | 254.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 232.60 | 254.35 | 254.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 236.80 | 234.89 | 242.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:30:00 | 237.50 | 234.89 | 242.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 242.78 | 233.52 | 240.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:00:00 | 242.78 | 233.52 | 240.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 242.63 | 233.61 | 240.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:15:00 | 243.11 | 233.61 | 240.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 240.40 | 233.72 | 240.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 230.98 | 234.19 | 240.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 244.24 | 234.70 | 240.29 | SL hit (close>static) qty=1.00 sl=243.63 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 280.89 | 244.78 | 244.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 292.59 | 246.31 | 245.53 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-25 11:45:00 | 123.71 | 2023-06-01 09:15:00 | 136.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-26 09:15:00 | 124.20 | 2023-06-01 09:15:00 | 136.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-09 09:30:00 | 291.70 | 2024-04-15 09:15:00 | 277.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 09:30:00 | 291.70 | 2024-04-18 09:15:00 | 293.70 | STOP_HIT | 0.50 | -0.69% |
| SELL | retest2 | 2024-04-18 09:30:00 | 293.83 | 2024-04-18 14:15:00 | 279.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 09:30:00 | 293.83 | 2024-04-23 09:15:00 | 287.27 | STOP_HIT | 0.50 | 2.23% |
| BUY | retest2 | 2024-09-26 14:30:00 | 247.11 | 2024-10-01 10:15:00 | 271.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-25 15:15:00 | 229.00 | 2025-03-28 09:15:00 | 236.98 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-03-26 09:30:00 | 227.66 | 2025-03-28 09:15:00 | 236.98 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2025-03-26 10:15:00 | 228.94 | 2025-03-28 09:15:00 | 236.98 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-04-01 15:00:00 | 230.13 | 2025-04-02 14:15:00 | 234.85 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-04-16 09:15:00 | 229.89 | 2025-04-16 12:15:00 | 236.59 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-04-16 10:15:00 | 230.14 | 2025-04-16 12:15:00 | 236.59 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-04-17 09:15:00 | 226.20 | 2025-04-17 11:15:00 | 236.72 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2025-04-30 15:15:00 | 230.52 | 2025-05-02 09:15:00 | 233.81 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-05-06 10:45:00 | 233.37 | 2025-05-07 14:15:00 | 236.23 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-05-07 09:30:00 | 233.40 | 2025-05-07 14:15:00 | 236.23 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-08 14:45:00 | 232.89 | 2025-05-12 09:15:00 | 243.47 | STOP_HIT | 1.00 | -4.54% |
| SELL | retest2 | 2025-08-18 11:45:00 | 268.47 | 2025-08-21 12:15:00 | 255.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 11:45:00 | 261.39 | 2025-08-25 09:15:00 | 248.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 11:45:00 | 268.47 | 2025-08-26 09:15:00 | 241.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 11:45:00 | 261.39 | 2025-08-26 11:15:00 | 235.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-11 15:00:00 | 257.85 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-12 09:15:00 | 258.75 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-12 11:15:00 | 257.86 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-12 11:45:00 | 257.76 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-12 13:45:00 | 258.60 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-15 10:00:00 | 257.78 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-15 11:45:00 | 258.19 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-15 13:00:00 | 257.80 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-22 14:45:00 | 258.80 | 2025-12-24 13:15:00 | 253.48 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-02-18 09:15:00 | 265.07 | 2026-02-25 12:15:00 | 242.20 | STOP_HIT | 1.00 | -8.63% |
| SELL | retest2 | 2026-04-02 09:15:00 | 230.98 | 2026-04-06 12:15:00 | 244.24 | STOP_HIT | 1.00 | -5.74% |
