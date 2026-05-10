# Railtel Corporation Of India Ltd. (RAILTEL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 343.35
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 9 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 9
- **Target hits / Stop hits / Partials:** 2 / 12 / 9
- **Avg / median % per leg:** 2.14% / 4.52%
- **Sum % (uncompounded):** 49.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 14 | 60.9% | 2 | 12 | 9 | 2.14% | 49.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 14 | 60.9% | 2 | 12 | 9 | 2.14% | 49.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 14 | 60.9% | 2 | 12 | 9 | 2.14% | 49.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 399.15 | 324.35 | 324.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 401.00 | 344.68 | 335.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 413.75 | 414.08 | 392.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 12:00:00 | 413.75 | 414.08 | 392.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 400.20 | 411.37 | 397.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 399.85 | 411.37 | 397.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 396.95 | 410.06 | 398.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 396.80 | 410.06 | 398.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 396.50 | 409.92 | 398.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 396.60 | 409.92 | 398.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 358.45 | 390.17 | 390.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 349.25 | 388.04 | 389.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 364.50 | 358.11 | 368.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 363.55 | 358.11 | 368.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 366.70 | 358.39 | 368.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:00:00 | 361.50 | 358.95 | 368.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 361.65 | 358.98 | 368.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 374.20 | 359.21 | 368.48 | SL hit (close>static) qty=1.00 sl=369.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 374.20 | 359.21 | 368.48 | SL hit (close>static) qty=1.00 sl=369.35 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 387.70 | 375.02 | 374.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 390.75 | 376.06 | 375.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 378.30 | 378.53 | 376.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 378.30 | 378.53 | 376.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 378.30 | 378.53 | 376.89 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 370.75 | 375.60 | 375.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 368.85 | 375.36 | 375.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 376.60 | 373.53 | 374.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 14:15:00 | 376.60 | 373.53 | 374.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 376.60 | 373.53 | 374.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 376.60 | 373.53 | 374.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 377.90 | 373.58 | 374.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 374.00 | 373.57 | 374.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 14:15:00 | 355.30 | 371.45 | 373.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 368.30 | 365.22 | 369.40 | SL hit (close>ema200) qty=0.50 sl=365.22 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 375.25 | 344.06 | 350.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 374.85 | 344.37 | 350.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:30:00 | 374.20 | 350.72 | 353.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 356.49 | 355.65 | 355.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 358.30 | 355.65 | 355.67 | SL hit (close>static) qty=0.50 sl=355.65 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 356.11 | 355.65 | 355.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 358.30 | 355.65 | 355.67 | SL hit (close>static) qty=0.50 sl=355.65 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 353.75 | 355.66 | 355.67 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 14:15:00 | 355.49 | 355.66 | 355.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 356.00 | 355.66 | 355.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 349.70 | 355.58 | 355.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 344.85 | 355.22 | 355.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-12 09:15:00 | 336.78 | 354.85 | 355.26 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 343.85 | 354.06 | 354.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 344.45 | 353.85 | 354.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 344.40 | 352.39 | 353.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 327.61 | 350.74 | 352.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 326.66 | 350.74 | 352.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 327.23 | 350.74 | 352.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 327.18 | 350.74 | 352.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 346.50 | 346.01 | 350.02 | SL hit (close>ema200) qty=0.50 sl=346.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 346.50 | 346.01 | 350.02 | SL hit (close>ema200) qty=0.50 sl=346.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 346.50 | 346.01 | 350.02 | SL hit (close>ema200) qty=0.50 sl=346.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 346.50 | 346.01 | 350.02 | SL hit (close>ema200) qty=0.50 sl=346.01 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 341.45 | 345.99 | 349.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 338.20 | 345.99 | 349.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 336.55 | 345.67 | 349.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 353.00 | 345.87 | 349.69 | SL hit (close>static) qty=1.00 sl=350.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 353.00 | 345.87 | 349.69 | SL hit (close>static) qty=1.00 sl=350.75 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 328.40 | 346.35 | 349.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 353.00 | 340.83 | 346.04 | SL hit (close>static) qty=1.00 sl=350.75 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 339.15 | 341.23 | 346.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 15:15:00 | 322.19 | 337.10 | 342.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 305.24 | 331.46 | 338.68 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 319.30 | 282.92 | 299.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 318.51 | 282.92 | 299.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 338.70 | 310.18 | 310.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 343.10 | 312.13 | 311.13 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-11 13:00:00 | 361.50 | 2025-09-12 09:15:00 | 374.20 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-09-11 13:30:00 | 361.65 | 2025-09-12 09:15:00 | 374.20 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-10-30 09:30:00 | 374.00 | 2025-11-06 14:15:00 | 355.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:30:00 | 374.00 | 2025-11-17 09:15:00 | 368.30 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2025-12-29 09:15:00 | 375.25 | 2026-01-08 11:15:00 | 356.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 09:15:00 | 375.25 | 2026-01-08 11:15:00 | 358.30 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-12-29 10:00:00 | 374.85 | 2026-01-08 11:15:00 | 356.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 10:00:00 | 374.85 | 2026-01-08 11:15:00 | 358.30 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2026-01-02 09:30:00 | 374.20 | 2026-01-08 14:15:00 | 355.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 09:30:00 | 374.20 | 2026-01-12 09:15:00 | 336.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-09 13:45:00 | 344.85 | 2026-01-21 09:15:00 | 327.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 343.85 | 2026-01-21 09:15:00 | 326.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:30:00 | 344.45 | 2026-01-21 09:15:00 | 327.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 12:00:00 | 344.40 | 2026-01-21 09:15:00 | 327.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 13:45:00 | 344.85 | 2026-01-28 14:15:00 | 346.50 | STOP_HIT | 0.50 | -0.48% |
| SELL | retest2 | 2026-01-13 09:15:00 | 343.85 | 2026-01-28 14:15:00 | 346.50 | STOP_HIT | 0.50 | -0.77% |
| SELL | retest2 | 2026-01-13 10:30:00 | 344.45 | 2026-01-28 14:15:00 | 346.50 | STOP_HIT | 0.50 | -0.60% |
| SELL | retest2 | 2026-01-19 12:00:00 | 344.40 | 2026-01-28 14:15:00 | 346.50 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2026-01-29 10:15:00 | 338.20 | 2026-01-30 13:15:00 | 353.00 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-01-30 09:15:00 | 336.55 | 2026-01-30 13:15:00 | 353.00 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-02-01 12:15:00 | 328.40 | 2026-02-10 09:15:00 | 353.00 | STOP_HIT | 1.00 | -7.49% |
| SELL | retest2 | 2026-02-11 09:15:00 | 339.15 | 2026-02-20 15:15:00 | 322.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 339.15 | 2026-03-02 09:15:00 | 305.24 | TARGET_HIT | 0.50 | 10.00% |
