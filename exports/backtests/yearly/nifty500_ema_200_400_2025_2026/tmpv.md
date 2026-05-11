# Tata Motors Passenger Vehicles Ltd. (TMPV)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 355.50
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
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 18 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 13
- **Target hits / Stop hits / Partials:** 6 / 13 / 7
- **Avg / median % per leg:** 2.77% / 5.00%
- **Sum % (uncompounded):** 72.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.65% | -6.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.69% | -4.7% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.63% | -1.9% |
| SELL (all) | 22 | 13 | 59.1% | 6 | 9 | 7 | 3.58% | 78.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 13 | 59.1% | 6 | 9 | 7 | 3.58% | 78.7% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.69% | -4.7% |
| retest2 (combined) | 25 | 13 | 52.0% | 6 | 12 | 7 | 3.07% | 76.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 442.82 | 412.70 | 412.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 444.94 | 418.24 | 415.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 428.48 | 429.12 | 423.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 15:00:00 | 433.18 | 429.14 | 423.44 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 412.85 | 429.01 | 423.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 412.85 | 429.01 | 423.43 | SL hit (close<ema400) qty=1.00 sl=423.43 alert=retest1 |

### Cycle 2 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 414.45 | 419.51 | 419.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 411.52 | 418.87 | 419.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 417.67 | 416.37 | 417.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 417.67 | 416.37 | 417.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 417.67 | 416.37 | 417.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 405.73 | 417.09 | 417.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:15:00 | 385.44 | 410.04 | 413.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 411.21 | 405.71 | 410.72 | SL hit (close>ema200) qty=0.50 sl=405.71 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 437.03 | 413.38 | 413.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 437.76 | 419.53 | 417.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 420.39 | 421.42 | 419.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 420.39 | 421.42 | 419.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 420.39 | 421.42 | 419.02 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 391.60 | 417.05 | 417.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 389.75 | 416.29 | 416.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 411.85 | 411.23 | 413.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 411.85 | 411.23 | 413.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 411.85 | 411.23 | 413.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 413.90 | 411.23 | 413.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 414.15 | 411.25 | 413.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 414.15 | 411.25 | 413.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 414.00 | 411.27 | 413.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 414.00 | 411.27 | 413.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 413.55 | 411.30 | 413.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 413.45 | 411.30 | 413.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 411.95 | 411.30 | 413.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:30:00 | 413.65 | 411.30 | 413.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 409.90 | 411.29 | 413.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:45:00 | 408.35 | 411.28 | 413.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:15:00 | 408.60 | 411.48 | 413.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:15:00 | 408.20 | 411.46 | 413.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:00:00 | 408.45 | 411.16 | 413.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 412.00 | 410.60 | 412.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 411.90 | 410.60 | 412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 403.15 | 410.53 | 412.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 402.10 | 409.93 | 412.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 402.05 | 409.86 | 412.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 14:15:00 | 387.93 | 408.16 | 411.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 14:15:00 | 388.17 | 408.16 | 411.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 14:15:00 | 387.79 | 408.16 | 411.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 14:15:00 | 388.03 | 408.16 | 411.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-17 09:15:00 | 367.52 | 407.68 | 410.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 384.50 | 367.53 | 367.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 387.00 | 371.08 | 369.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 373.05 | 373.21 | 370.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 12:15:00 | 369.85 | 373.15 | 370.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 369.85 | 373.15 | 370.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:00:00 | 369.85 | 373.15 | 370.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 371.10 | 373.13 | 370.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 366.55 | 373.13 | 370.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 371.15 | 373.11 | 370.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 372.15 | 373.11 | 370.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 371.20 | 373.09 | 370.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 357.65 | 373.09 | 370.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 356.90 | 372.93 | 370.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 356.90 | 372.93 | 370.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 331.95 | 368.39 | 368.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 328.90 | 367.64 | 368.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 330.90 | 326.70 | 341.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 330.90 | 326.70 | 341.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 341.60 | 328.29 | 341.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 13:30:00 | 341.55 | 328.29 | 341.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 342.35 | 328.43 | 341.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:00:00 | 342.35 | 328.43 | 341.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 343.00 | 328.58 | 341.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 337.80 | 328.58 | 341.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 10:15:00 | 344.25 | 328.84 | 341.24 | SL hit (close>static) qty=1.00 sl=343.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-13 15:00:00 | 433.18 | 2025-06-16 09:15:00 | 412.85 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2025-06-16 13:30:00 | 416.00 | 2025-06-17 09:15:00 | 411.03 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-27 09:15:00 | 416.36 | 2025-07-01 14:15:00 | 414.45 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-07-01 12:15:00 | 415.45 | 2025-07-01 14:15:00 | 414.45 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-30 09:15:00 | 405.73 | 2025-08-07 11:15:00 | 385.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 09:15:00 | 405.73 | 2025-08-18 09:15:00 | 411.21 | STOP_HIT | 0.50 | -1.35% |
| SELL | retest2 | 2025-08-18 15:00:00 | 409.64 | 2025-08-19 10:15:00 | 423.21 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-08-28 09:15:00 | 409.09 | 2025-09-01 12:15:00 | 414.48 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-28 15:00:00 | 409.06 | 2025-09-02 11:15:00 | 420.33 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-08-29 15:00:00 | 405.33 | 2025-09-02 11:15:00 | 420.33 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-10-30 10:45:00 | 408.35 | 2025-11-14 14:15:00 | 387.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 12:15:00 | 408.60 | 2025-11-14 14:15:00 | 388.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 13:15:00 | 408.20 | 2025-11-14 14:15:00 | 387.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 13:00:00 | 408.45 | 2025-11-14 14:15:00 | 388.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 10:45:00 | 408.35 | 2025-11-17 09:15:00 | 367.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 12:15:00 | 408.60 | 2025-11-17 09:15:00 | 367.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 13:15:00 | 408.20 | 2025-11-17 09:15:00 | 367.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 13:00:00 | 408.45 | 2025-11-17 09:15:00 | 367.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 402.10 | 2025-11-17 09:15:00 | 382.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 15:00:00 | 402.05 | 2025-11-17 09:15:00 | 381.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 402.10 | 2025-11-19 13:15:00 | 361.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 15:00:00 | 402.05 | 2025-11-19 13:15:00 | 361.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 337.80 | 2026-04-13 10:15:00 | 344.25 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-04-30 11:30:00 | 341.90 | 2026-05-04 10:15:00 | 343.95 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-04-30 15:00:00 | 341.25 | 2026-05-04 10:15:00 | 343.95 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-05-04 12:00:00 | 342.05 | 2026-05-04 12:15:00 | 343.95 | STOP_HIT | 1.00 | -0.56% |
