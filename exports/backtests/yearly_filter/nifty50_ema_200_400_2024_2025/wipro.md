# WIPRO (WIPRO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 197.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 15 / 21
- **Target hits / Stop hits / Partials:** 1 / 28 / 7
- **Avg / median % per leg:** 0.63% / -0.81%
- **Sum % (uncompounded):** 22.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 1 | 9.1% | 1 | 10 | 0 | -0.73% | -8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 1 | 10 | 0 | -0.73% | -8.0% |
| SELL (all) | 25 | 14 | 56.0% | 0 | 18 | 7 | 1.23% | 30.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 14 | 56.0% | 0 | 18 | 7 | 1.23% | 30.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 15 | 41.7% | 1 | 28 | 7 | 0.63% | 22.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 249.00 | 235.18 | 235.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 253.25 | 238.18 | 236.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 09:15:00 | 255.38 | 261.21 | 251.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:45:00 | 255.90 | 261.21 | 251.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 251.68 | 260.99 | 251.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 251.68 | 260.99 | 251.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 252.60 | 260.91 | 251.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 256.65 | 258.93 | 251.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:30:00 | 254.00 | 259.46 | 253.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 251.00 | 259.38 | 253.17 | SL hit (close<static) qty=1.00 sl=251.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 13:15:00 | 286.35 | 299.36 | 299.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 283.25 | 298.69 | 299.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 250.81 | 249.03 | 261.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 10:00:00 | 250.81 | 249.03 | 261.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 254.80 | 249.45 | 255.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 254.76 | 249.45 | 255.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 254.97 | 249.51 | 255.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 255.33 | 249.51 | 255.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 254.85 | 249.61 | 255.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 254.85 | 249.61 | 255.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 255.02 | 249.72 | 255.27 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 265.06 | 258.50 | 258.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 266.11 | 258.92 | 258.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 260.45 | 262.53 | 260.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 260.45 | 262.53 | 260.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 260.45 | 262.53 | 260.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 262.25 | 261.39 | 260.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 262.15 | 261.40 | 260.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 262.15 | 261.43 | 260.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 268.80 | 261.40 | 260.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 261.10 | 261.77 | 260.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 260.55 | 261.77 | 260.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 261.75 | 261.78 | 260.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 261.00 | 261.78 | 260.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 260.25 | 261.76 | 260.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 260.25 | 261.76 | 260.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 260.10 | 261.74 | 260.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 259.55 | 261.74 | 260.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 259.50 | 261.71 | 260.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 259.50 | 261.71 | 260.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 260.85 | 261.71 | 260.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 261.60 | 261.52 | 260.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 261.65 | 261.52 | 260.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:30:00 | 261.30 | 261.51 | 260.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 261.15 | 261.51 | 260.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 261.50 | 261.51 | 260.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:30:00 | 260.95 | 261.51 | 260.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 257.80 | 261.48 | 260.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 257.80 | 261.48 | 260.68 | SL hit (close<static) qty=1.00 sl=258.45 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 251.35 | 259.90 | 259.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 250.20 | 259.81 | 259.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 251.85 | 250.05 | 253.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 251.85 | 250.05 | 253.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 254.69 | 250.11 | 253.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 255.48 | 250.11 | 253.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 256.20 | 250.17 | 253.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 256.20 | 250.17 | 253.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 254.40 | 250.38 | 253.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 253.11 | 250.38 | 253.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 253.16 | 250.43 | 253.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 255.20 | 249.40 | 252.05 | SL hit (close>static) qty=1.00 sl=254.55 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 257.29 | 246.72 | 246.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.80 | 257.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 11:00:00 | 261.40 | 261.80 | 257.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.98 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 235.55 | 254.63 | 254.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 233.70 | 249.46 | 251.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 202.49 | 199.74 | 212.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:00:00 | 202.49 | 199.74 | 212.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 209.46 | 201.37 | 210.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:15:00 | 209.22 | 201.37 | 210.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:45:00 | 209.20 | 201.45 | 210.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 209.18 | 201.53 | 210.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 204.19 | 201.86 | 210.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 198.76 | 202.42 | 209.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 198.74 | 202.42 | 209.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 198.72 | 202.42 | 209.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 203.61 | 202.24 | 209.42 | SL hit (close>ema200) qty=0.50 sl=202.24 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-26 09:15:00 | 256.65 | 2024-08-02 14:15:00 | 251.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-08-02 13:30:00 | 254.00 | 2024-08-02 14:15:00 | 251.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-16 10:00:00 | 254.13 | 2024-10-21 09:15:00 | 279.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-16 12:30:00 | 262.25 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-07-16 13:45:00 | 262.15 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-17 11:30:00 | 262.15 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-18 09:15:00 | 268.80 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2025-07-23 15:00:00 | 261.60 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-24 09:15:00 | 261.65 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-07-24 12:30:00 | 261.30 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-24 13:00:00 | 261.15 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-26 09:15:00 | 253.11 | 2025-09-10 09:15:00 | 255.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-08-26 10:30:00 | 253.16 | 2025-09-10 09:15:00 | 255.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-11 09:15:00 | 253.10 | 2025-09-17 10:15:00 | 254.96 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-11 09:45:00 | 253.09 | 2025-09-17 10:15:00 | 254.96 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-22 10:45:00 | 250.15 | 2025-09-26 09:15:00 | 237.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:45:00 | 250.11 | 2025-09-26 09:15:00 | 237.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 250.00 | 2025-09-26 09:15:00 | 237.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:45:00 | 250.00 | 2025-09-26 09:15:00 | 237.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:45:00 | 250.15 | 2025-10-09 14:15:00 | 246.49 | STOP_HIT | 0.50 | 1.46% |
| SELL | retest2 | 2025-09-22 11:45:00 | 250.11 | 2025-10-09 14:15:00 | 246.49 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2025-09-22 13:45:00 | 250.00 | 2025-10-09 14:15:00 | 246.49 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-09-22 14:45:00 | 250.00 | 2025-10-09 14:15:00 | 246.49 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-10-10 11:15:00 | 249.65 | 2025-10-16 11:15:00 | 252.74 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-10 12:00:00 | 249.66 | 2025-10-16 11:15:00 | 252.74 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-10 13:00:00 | 249.60 | 2025-10-16 11:15:00 | 252.74 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-10-15 09:30:00 | 249.69 | 2025-10-16 11:15:00 | 252.74 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-17 09:15:00 | 242.98 | 2025-12-03 09:15:00 | 254.90 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2025-11-24 11:00:00 | 248.46 | 2025-12-03 09:15:00 | 254.90 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-11-27 12:15:00 | 248.71 | 2025-12-03 09:15:00 | 254.90 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2026-04-16 10:15:00 | 209.22 | 2026-04-24 09:15:00 | 198.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 10:45:00 | 209.20 | 2026-04-24 09:15:00 | 198.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 11:30:00 | 209.18 | 2026-04-24 09:15:00 | 198.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 10:15:00 | 209.22 | 2026-04-27 09:15:00 | 203.61 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2026-04-16 10:45:00 | 209.20 | 2026-04-27 09:15:00 | 203.61 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2026-04-16 11:30:00 | 209.18 | 2026-04-27 09:15:00 | 203.61 | STOP_HIT | 0.50 | 2.66% |
