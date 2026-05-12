# RBL Bank Ltd. (RBLBANK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 343.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 27
- **Target hits / Stop hits / Partials:** 1 / 33 / 9
- **Avg / median % per leg:** 0.24% / -1.34%
- **Sum % (uncompounded):** 10.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.07% | -29.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.07% | -29.0% |
| SELL (all) | 29 | 16 | 55.2% | 1 | 19 | 9 | 1.35% | 39.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 16 | 55.2% | 1 | 19 | 9 | 1.35% | 39.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 16 | 37.2% | 1 | 33 | 9 | 0.24% | 10.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 251.85 | 262.09 | 262.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 10:15:00 | 240.85 | 260.93 | 261.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 248.15 | 247.26 | 252.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 15:00:00 | 248.15 | 247.26 | 252.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 252.65 | 247.60 | 252.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 15:00:00 | 252.65 | 247.60 | 252.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 253.30 | 247.65 | 252.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:15:00 | 263.35 | 247.65 | 252.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 256.60 | 247.74 | 252.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 11:00:00 | 254.90 | 247.81 | 252.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 09:30:00 | 255.15 | 248.21 | 252.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 12:45:00 | 255.05 | 248.42 | 252.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 13:15:00 | 255.25 | 248.90 | 252.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 253.70 | 251.10 | 253.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 11:00:00 | 253.70 | 251.10 | 253.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 255.20 | 251.14 | 253.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 12:00:00 | 255.20 | 251.14 | 253.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 252.60 | 251.17 | 253.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 15:00:00 | 250.85 | 251.17 | 253.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 248.35 | 251.17 | 253.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 14:15:00 | 242.16 | 250.88 | 253.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 14:15:00 | 242.39 | 250.88 | 253.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 14:15:00 | 242.30 | 250.88 | 253.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 14:15:00 | 242.49 | 250.88 | 253.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 12:15:00 | 251.45 | 250.77 | 253.23 | SL hit (close>ema200) qty=0.50 sl=250.77 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 15:15:00 | 260.00 | 254.91 | 254.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 09:15:00 | 267.45 | 255.03 | 254.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 12:15:00 | 256.05 | 256.47 | 255.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-06 13:00:00 | 256.05 | 256.47 | 255.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 13:15:00 | 256.20 | 256.47 | 255.72 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 245.50 | 255.03 | 255.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 243.40 | 254.91 | 254.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 252.75 | 252.44 | 253.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 12:00:00 | 252.75 | 252.44 | 253.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 252.30 | 252.37 | 253.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 14:45:00 | 250.40 | 252.35 | 253.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 09:30:00 | 249.90 | 252.32 | 253.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 10:15:00 | 251.20 | 252.32 | 253.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 11:00:00 | 251.05 | 252.31 | 253.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 252.55 | 252.24 | 253.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 251.15 | 252.24 | 253.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:45:00 | 250.70 | 252.22 | 253.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 13:15:00 | 254.25 | 252.25 | 253.32 | SL hit (close>static) qty=1.00 sl=253.75 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 268.93 | 252.92 | 252.92 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 246.80 | 254.51 | 254.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 245.65 | 254.35 | 254.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 13:15:00 | 227.21 | 227.11 | 236.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 14:00:00 | 227.21 | 227.11 | 236.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 177.99 | 168.85 | 180.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 177.18 | 169.50 | 180.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:15:00 | 168.32 | 170.20 | 179.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 170.78 | 170.10 | 179.00 | SL hit (close>ema200) qty=0.50 sl=170.10 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 14:15:00 | 179.21 | 163.54 | 163.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 180.84 | 168.25 | 166.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 252.95 | 256.21 | 243.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 252.95 | 256.21 | 243.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 246.55 | 256.06 | 246.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 246.55 | 256.06 | 246.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 247.15 | 255.97 | 246.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 247.15 | 255.97 | 246.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 299.95 | 310.75 | 300.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 299.95 | 310.75 | 300.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 301.80 | 310.66 | 300.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 300.10 | 310.66 | 300.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 299.30 | 310.01 | 300.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 299.30 | 310.01 | 300.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 298.90 | 309.90 | 300.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 297.60 | 309.90 | 300.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 300.50 | 309.37 | 300.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 300.10 | 309.37 | 300.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 300.40 | 309.28 | 300.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 13:15:00 | 304.65 | 309.20 | 300.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 303.00 | 308.96 | 300.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 299.30 | 308.78 | 300.19 | SL hit (close<static) qty=1.00 sl=299.80 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 290.70 | 302.67 | 302.72 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 306.15 | 302.77 | 302.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 307.55 | 302.81 | 302.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 313.20 | 314.88 | 310.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 10:45:00 | 313.50 | 314.88 | 310.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 309.80 | 314.82 | 310.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:00:00 | 309.80 | 314.82 | 310.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 308.70 | 314.76 | 310.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:00:00 | 308.70 | 314.76 | 310.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 312.05 | 314.74 | 310.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 308.25 | 314.74 | 310.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 306.90 | 314.62 | 310.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 306.90 | 314.62 | 310.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 309.70 | 314.05 | 309.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 309.70 | 314.05 | 309.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 308.20 | 313.99 | 309.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 308.20 | 313.99 | 309.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 307.55 | 313.93 | 309.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 307.30 | 313.93 | 309.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 309.50 | 313.84 | 309.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 310.00 | 313.84 | 309.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 308.05 | 313.78 | 309.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 308.05 | 313.78 | 309.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 307.30 | 313.71 | 309.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 306.60 | 313.71 | 309.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 307.05 | 311.58 | 309.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 307.30 | 311.58 | 309.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 303.75 | 311.50 | 309.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 303.75 | 311.50 | 309.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 11:15:00 | 299.45 | 307.12 | 307.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 293.95 | 306.75 | 306.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 307.45 | 303.80 | 305.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 10:15:00 | 307.45 | 303.80 | 305.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 307.45 | 303.80 | 305.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:45:00 | 307.10 | 303.80 | 305.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 307.05 | 303.84 | 305.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:00:00 | 307.05 | 303.84 | 305.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 305.70 | 303.85 | 305.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:30:00 | 305.30 | 303.88 | 305.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 290.04 | 303.28 | 304.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 303.05 | 302.55 | 304.52 | SL hit (close>ema200) qty=0.50 sl=302.55 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 326.00 | 306.13 | 306.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 337.00 | 312.91 | 310.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-04 11:00:00 | 254.90 | 2024-04-16 14:15:00 | 242.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 09:30:00 | 255.15 | 2024-04-16 14:15:00 | 242.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 12:45:00 | 255.05 | 2024-04-16 14:15:00 | 242.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 13:15:00 | 255.25 | 2024-04-16 14:15:00 | 242.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-04 11:00:00 | 254.90 | 2024-04-18 12:15:00 | 251.45 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2024-04-05 09:30:00 | 255.15 | 2024-04-18 12:15:00 | 251.45 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2024-04-05 12:45:00 | 255.05 | 2024-04-18 12:15:00 | 251.45 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2024-04-08 13:15:00 | 255.25 | 2024-04-18 12:15:00 | 251.45 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2024-04-15 15:00:00 | 250.85 | 2024-04-19 09:15:00 | 238.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-16 09:15:00 | 248.35 | 2024-04-19 09:15:00 | 238.07 | PARTIAL | 0.50 | 4.14% |
| SELL | retest2 | 2024-04-15 15:00:00 | 250.85 | 2024-04-19 13:15:00 | 252.40 | STOP_HIT | 0.50 | -0.62% |
| SELL | retest2 | 2024-04-16 09:15:00 | 248.35 | 2024-04-19 13:15:00 | 252.40 | STOP_HIT | 0.50 | -1.63% |
| SELL | retest2 | 2024-04-18 13:30:00 | 250.60 | 2024-04-19 14:15:00 | 254.25 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-05-21 14:45:00 | 250.40 | 2024-05-23 13:15:00 | 254.25 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-05-22 09:30:00 | 249.90 | 2024-05-23 13:15:00 | 254.25 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-05-22 10:15:00 | 251.20 | 2024-05-23 13:15:00 | 254.25 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-05-22 11:00:00 | 251.05 | 2024-05-23 13:15:00 | 254.25 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-05-23 11:15:00 | 251.15 | 2024-05-24 12:15:00 | 255.35 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-05-23 11:45:00 | 250.70 | 2024-05-24 12:15:00 | 255.35 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-05-28 10:00:00 | 251.25 | 2024-06-03 09:15:00 | 257.15 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-05-28 11:15:00 | 251.05 | 2024-06-03 09:15:00 | 257.15 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-06-04 09:15:00 | 247.50 | 2024-06-04 11:15:00 | 235.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 247.50 | 2024-06-04 14:15:00 | 222.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-10 14:30:00 | 253.00 | 2024-06-18 09:15:00 | 262.34 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-06-11 15:00:00 | 253.31 | 2024-06-18 09:15:00 | 262.34 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-12-12 09:30:00 | 177.18 | 2024-12-17 11:15:00 | 168.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:30:00 | 177.18 | 2024-12-18 09:15:00 | 170.78 | STOP_HIT | 0.50 | 3.61% |
| BUY | retest2 | 2025-12-05 13:15:00 | 304.65 | 2025-12-08 12:15:00 | 299.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-12-08 10:30:00 | 303.00 | 2025-12-08 12:15:00 | 299.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-09 11:30:00 | 302.90 | 2025-12-16 11:15:00 | 298.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-09 13:00:00 | 302.85 | 2025-12-16 11:15:00 | 298.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-22 11:30:00 | 304.25 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-12-22 13:00:00 | 303.70 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-12-22 15:15:00 | 303.80 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-12-23 09:30:00 | 305.00 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-12-30 09:30:00 | 308.25 | 2026-01-19 09:15:00 | 303.75 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-09 14:45:00 | 307.45 | 2026-01-19 09:15:00 | 303.75 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-12 10:00:00 | 306.20 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2026-01-12 15:15:00 | 306.20 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2026-01-14 10:45:00 | 308.45 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2026-01-14 12:15:00 | 308.25 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2026-03-25 14:30:00 | 305.30 | 2026-03-30 09:15:00 | 290.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:30:00 | 305.30 | 2026-04-01 12:15:00 | 303.05 | STOP_HIT | 0.50 | 0.74% |
