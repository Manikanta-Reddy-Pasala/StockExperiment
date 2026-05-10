# Brainbees Solutions Ltd. (FIRSTCRY)

## Backtest Summary

- **Window:** 2024-08-13 09:15:00 → 2026-05-08 15:15:00 (3000 bars)
- **Last close:** 234.91
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
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 5 / 17
- **Target hits / Stop hits / Partials:** 0 / 19 / 3
- **Avg / median % per leg:** -2.15% / -2.75%
- **Sum % (uncompounded):** -47.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.94% | -11.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.94% | -11.8% |
| SELL (all) | 19 | 5 | 26.3% | 0 | 16 | 3 | -1.86% | -35.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 5 | 26.3% | 0 | 16 | 3 | -1.86% | -35.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 5 | 22.7% | 0 | 19 | 3 | -2.15% | -47.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 12:15:00 | 380.40 | 372.19 | 372.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 382.35 | 372.29 | 372.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 372.15 | 373.35 | 372.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 372.15 | 373.35 | 372.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 372.15 | 373.35 | 372.77 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 362.25 | 372.19 | 372.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 358.25 | 371.86 | 372.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 364.10 | 363.73 | 367.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 364.10 | 363.73 | 367.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 364.10 | 363.73 | 367.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:45:00 | 364.30 | 363.73 | 367.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 369.55 | 363.81 | 367.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 362.05 | 364.08 | 367.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 360.65 | 364.00 | 367.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:45:00 | 362.80 | 363.57 | 366.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 373.85 | 363.95 | 366.90 | SL hit (close>static) qty=1.00 sl=371.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 373.85 | 363.95 | 366.90 | SL hit (close>static) qty=1.00 sl=371.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 373.85 | 363.95 | 366.90 | SL hit (close>static) qty=1.00 sl=371.80 alert=retest2 |

### Cycle 3 — BUY (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 15:15:00 | 389.00 | 369.21 | 369.14 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 14:15:00 | 355.70 | 369.18 | 369.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 352.35 | 367.74 | 368.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 373.75 | 367.51 | 368.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 373.75 | 367.51 | 368.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 373.75 | 367.51 | 368.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 378.60 | 367.51 | 368.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 390.00 | 367.73 | 368.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 390.00 | 367.73 | 368.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 399.90 | 369.23 | 369.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 413.20 | 369.67 | 369.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 379.30 | 382.09 | 377.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 379.35 | 382.09 | 377.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 380.25 | 382.05 | 377.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 377.85 | 382.05 | 377.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 375.60 | 381.99 | 377.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 375.60 | 381.99 | 377.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 385.75 | 382.03 | 377.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:30:00 | 386.95 | 382.07 | 377.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 389.85 | 382.07 | 377.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 387.05 | 382.17 | 377.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 372.65 | 382.04 | 377.57 | SL hit (close<static) qty=1.00 sl=374.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 372.65 | 382.04 | 377.57 | SL hit (close<static) qty=1.00 sl=374.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 372.65 | 382.04 | 377.57 | SL hit (close<static) qty=1.00 sl=374.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 362.70 | 374.62 | 374.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 359.20 | 371.72 | 373.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 12:15:00 | 298.60 | 298.47 | 315.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:30:00 | 298.60 | 298.47 | 315.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 249.86 | 229.11 | 249.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 249.86 | 229.11 | 249.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 247.93 | 229.30 | 249.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 236.30 | 230.15 | 249.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 224.49 | 230.07 | 248.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 231.70 | 229.80 | 248.01 | SL hit (close>ema200) qty=0.50 sl=229.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 245.20 | 227.97 | 242.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:00:00 | 245.10 | 228.43 | 242.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:15:00 | 246.42 | 228.62 | 242.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 243.65 | 229.68 | 242.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:00:00 | 243.65 | 229.68 | 242.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 242.75 | 229.81 | 242.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:30:00 | 243.93 | 229.81 | 242.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 242.54 | 229.93 | 242.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 241.31 | 229.93 | 242.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:45:00 | 241.20 | 230.05 | 242.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 245.50 | 230.45 | 242.93 | SL hit (close>static) qty=1.00 sl=243.13 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 245.50 | 230.45 | 242.93 | SL hit (close>static) qty=1.00 sl=243.13 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 238.20 | 231.36 | 243.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 243.28 | 231.88 | 243.00 | SL hit (close>static) qty=1.00 sl=243.13 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 259.21 | 234.45 | 243.44 | SL hit (close>static) qty=1.00 sl=252.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 259.21 | 234.45 | 243.44 | SL hit (close>static) qty=1.00 sl=252.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 259.21 | 234.45 | 243.44 | SL hit (close>static) qty=1.00 sl=252.07 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 15:00:00 | 241.30 | 241.31 | 245.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 238.37 | 240.17 | 243.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 237.38 | 240.11 | 243.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 13:45:00 | 371.45 | 2025-05-26 15:15:00 | 376.55 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-27 09:15:00 | 366.90 | 2025-05-30 10:15:00 | 348.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-27 09:15:00 | 366.90 | 2025-06-03 10:15:00 | 368.30 | STOP_HIT | 0.50 | -0.38% |
| SELL | retest2 | 2025-06-03 11:00:00 | 368.30 | 2025-06-03 12:15:00 | 388.10 | STOP_HIT | 1.00 | -5.38% |
| SELL | retest2 | 2025-06-03 15:15:00 | 362.10 | 2025-06-05 09:15:00 | 396.40 | STOP_HIT | 1.00 | -9.47% |
| SELL | retest2 | 2025-06-19 10:00:00 | 361.80 | 2025-06-19 15:15:00 | 343.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 10:00:00 | 361.80 | 2025-06-26 10:15:00 | 362.80 | STOP_HIT | 0.50 | -0.28% |
| SELL | retest2 | 2025-07-04 14:15:00 | 361.65 | 2025-07-07 13:15:00 | 371.60 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-08-05 14:30:00 | 362.05 | 2025-08-12 09:15:00 | 373.85 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-08-06 09:30:00 | 360.65 | 2025-08-12 09:15:00 | 373.85 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2025-08-11 09:45:00 | 362.80 | 2025-08-12 09:15:00 | 373.85 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-09-23 12:30:00 | 386.95 | 2025-09-26 09:15:00 | 372.65 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-09-23 13:15:00 | 389.85 | 2025-09-26 09:15:00 | 372.65 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-09-24 15:15:00 | 387.05 | 2025-09-26 09:15:00 | 372.65 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-03-23 09:15:00 | 236.30 | 2026-03-23 12:15:00 | 224.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 236.30 | 2026-03-24 12:15:00 | 231.70 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2026-04-07 14:15:00 | 245.20 | 2026-04-10 09:15:00 | 245.50 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2026-04-08 10:00:00 | 245.10 | 2026-04-10 09:15:00 | 245.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-04-08 11:15:00 | 246.42 | 2026-04-13 13:15:00 | 243.28 | STOP_HIT | 1.00 | 1.27% |
| SELL | retest2 | 2026-04-09 13:15:00 | 241.31 | 2026-04-17 09:15:00 | 259.21 | STOP_HIT | 1.00 | -7.42% |
| SELL | retest2 | 2026-04-09 13:45:00 | 241.20 | 2026-04-17 09:15:00 | 259.21 | STOP_HIT | 1.00 | -7.47% |
| SELL | retest2 | 2026-04-13 09:15:00 | 238.20 | 2026-04-17 09:15:00 | 259.21 | STOP_HIT | 1.00 | -8.82% |
