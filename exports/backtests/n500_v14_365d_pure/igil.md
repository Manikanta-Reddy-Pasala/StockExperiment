# International Gemmological Institute (India) Ltd. (IGIL)

## Backtest Summary

- **Window:** 2024-12-20 09:15:00 → 2026-05-08 15:15:00 (2382 bars)
- **Last close:** 352.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 6 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 2
- **Avg / median % per leg:** -0.70% / -3.11%
- **Sum % (uncompounded):** -11.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.77% | -15.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.77% | -15.1% |
| SELL (all) | 13 | 4 | 30.8% | 2 | 9 | 2 | 0.24% | 3.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 4 | 30.8% | 2 | 9 | 2 | 0.24% | 3.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 4 | 23.5% | 2 | 13 | 2 | -0.70% | -12.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 394.55 | 383.20 | 383.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 15:15:00 | 395.05 | 383.43 | 383.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 385.70 | 391.87 | 388.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 385.70 | 391.87 | 388.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 385.70 | 391.87 | 388.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 385.70 | 391.87 | 388.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 384.35 | 391.79 | 388.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 384.00 | 391.79 | 388.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 349.50 | 384.86 | 384.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 348.90 | 384.50 | 384.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 374.40 | 354.35 | 363.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 374.40 | 354.35 | 363.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 374.40 | 354.35 | 363.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 374.40 | 354.35 | 363.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 376.55 | 354.57 | 363.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 376.55 | 354.57 | 363.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 364.00 | 356.26 | 364.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 374.00 | 356.26 | 364.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 372.35 | 356.42 | 364.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 372.20 | 356.42 | 364.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 370.00 | 356.56 | 364.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 366.30 | 356.69 | 364.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 366.65 | 356.81 | 364.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:00:00 | 366.05 | 356.90 | 364.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 368.15 | 357.16 | 364.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 373.30 | 357.43 | 364.56 | SL hit (close>static) qty=1.00 sl=372.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 373.30 | 357.43 | 364.56 | SL hit (close>static) qty=1.00 sl=372.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 373.30 | 357.43 | 364.56 | SL hit (close>static) qty=1.00 sl=372.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 373.30 | 357.43 | 364.56 | SL hit (close>static) qty=1.00 sl=372.25 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 387.00 | 359.77 | 365.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 376.55 | 363.16 | 366.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 376.55 | 363.28 | 366.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 357.72 | 363.86 | 366.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 357.72 | 363.86 | 366.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-14 09:15:00 | 338.90 | 355.56 | 360.74 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-14 09:15:00 | 338.90 | 355.56 | 360.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 13:15:00 | 330.00 | 325.37 | 325.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 15:15:00 | 331.00 | 325.47 | 325.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 325.00 | 325.47 | 325.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 325.00 | 325.47 | 325.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 325.00 | 325.47 | 325.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 325.80 | 325.46 | 325.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 12:15:00 | 327.00 | 325.46 | 325.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:00:00 | 326.85 | 325.49 | 325.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 15:15:00 | 325.75 | 325.48 | 325.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 325.75 | 325.49 | 325.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 311.05 | 325.49 | 325.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 314.05 | 325.37 | 325.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 314.05 | 325.37 | 325.36 | SL hit (close<static) qty=1.00 sl=316.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 314.05 | 325.37 | 325.36 | SL hit (close<static) qty=1.00 sl=316.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 314.05 | 325.37 | 325.36 | SL hit (close<static) qty=1.00 sl=316.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 314.05 | 325.37 | 325.36 | SL hit (close<static) qty=1.00 sl=316.30 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 317.15 | 325.29 | 325.32 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 329.15 | 325.35 | 325.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 330.90 | 325.44 | 325.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 320.50 | 326.52 | 325.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 320.50 | 326.52 | 325.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 320.50 | 326.52 | 325.97 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 303.70 | 325.31 | 325.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 299.20 | 324.89 | 325.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 324.10 | 323.28 | 324.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 14:15:00 | 324.10 | 323.28 | 324.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 324.10 | 323.28 | 324.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 327.25 | 323.28 | 324.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 318.95 | 323.24 | 324.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 312.70 | 323.24 | 324.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 10:45:00 | 317.30 | 323.14 | 324.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 327.60 | 323.17 | 324.23 | SL hit (close>static) qty=1.00 sl=326.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 327.60 | 323.17 | 324.23 | SL hit (close>static) qty=1.00 sl=326.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 15:15:00 | 316.95 | 324.42 | 324.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 326.80 | 324.37 | 324.76 | SL hit (close>static) qty=1.00 sl=326.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 330.95 | 325.18 | 325.15 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 321.25 | 325.12 | 325.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 320.30 | 325.03 | 325.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 327.65 | 324.35 | 324.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 327.65 | 324.35 | 324.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 327.65 | 324.35 | 324.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 317.05 | 324.42 | 324.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 321.75 | 323.74 | 324.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 12:15:00 | 333.25 | 323.60 | 324.26 | SL hit (close>static) qty=1.00 sl=332.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 12:15:00 | 333.25 | 323.60 | 324.26 | SL hit (close>static) qty=1.00 sl=332.45 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 10:15:00 | 340.15 | 325.04 | 324.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 346.85 | 326.74 | 325.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 342.80 | 343.11 | 336.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:30:00 | 343.50 | 343.11 | 336.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-12 12:15:00 | 366.30 | 2025-09-15 10:15:00 | 373.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-09-12 13:15:00 | 366.65 | 2025-09-15 10:15:00 | 373.30 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-12 14:00:00 | 366.05 | 2025-09-15 10:15:00 | 373.30 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-09-15 09:15:00 | 368.15 | 2025-09-15 10:15:00 | 373.30 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-19 12:00:00 | 376.55 | 2025-09-25 11:15:00 | 357.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:30:00 | 376.55 | 2025-09-25 11:15:00 | 357.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:00:00 | 376.55 | 2025-10-14 09:15:00 | 338.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-19 12:30:00 | 376.55 | 2025-10-14 09:15:00 | 338.90 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-02 11:30:00 | 325.80 | 2026-03-04 09:15:00 | 314.05 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-03-02 12:15:00 | 327.00 | 2026-03-04 09:15:00 | 314.05 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2026-03-02 14:00:00 | 326.85 | 2026-03-04 09:15:00 | 314.05 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2026-03-02 15:15:00 | 325.75 | 2026-03-04 09:15:00 | 314.05 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-03-18 09:15:00 | 312.70 | 2026-03-18 13:15:00 | 327.60 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2026-03-18 10:45:00 | 317.30 | 2026-03-18 13:15:00 | 327.60 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-03-23 15:15:00 | 316.95 | 2026-03-24 09:15:00 | 326.80 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-04-02 09:15:00 | 317.05 | 2026-04-08 12:15:00 | 333.25 | STOP_HIT | 1.00 | -5.11% |
| SELL | retest2 | 2026-04-07 09:15:00 | 321.75 | 2026-04-08 12:15:00 | 333.25 | STOP_HIT | 1.00 | -3.57% |
