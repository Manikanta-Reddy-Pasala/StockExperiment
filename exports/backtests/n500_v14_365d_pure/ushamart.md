# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 472.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 15
- **Target hits / Stop hits / Partials:** 2 / 17 / 5
- **Avg / median % per leg:** -0.37% / -1.92%
- **Sum % (uncompounded):** -8.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 2 | 1 | 0 | 5.90% | 17.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 2 | 1 | 0 | 5.90% | 17.7% |
| SELL (all) | 21 | 7 | 33.3% | 0 | 16 | 5 | -1.26% | -26.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 7 | 33.3% | 0 | 16 | 5 | -1.26% | -26.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 9 | 37.5% | 2 | 17 | 5 | -0.37% | -8.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 346.30 | 316.49 | 316.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 351.00 | 316.84 | 316.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 360.60 | 362.75 | 347.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 360.60 | 362.75 | 347.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 352.10 | 366.00 | 352.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 352.10 | 366.00 | 352.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 349.20 | 365.83 | 352.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 350.75 | 365.83 | 352.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 347.20 | 364.70 | 352.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 347.20 | 364.70 | 352.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 351.30 | 361.04 | 351.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 352.60 | 359.85 | 351.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 344.45 | 359.49 | 351.66 | SL hit (close<static) qty=1.00 sl=346.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 354.55 | 359.29 | 351.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 11:15:00 | 352.25 | 359.05 | 351.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-25 13:15:00 | 387.48 | 363.20 | 355.13 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-26 09:15:00 | 390.01 | 363.95 | 355.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 421.40 | 438.71 | 438.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 414.30 | 437.88 | 438.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 431.95 | 423.88 | 429.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 10:15:00 | 431.95 | 423.88 | 429.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 431.95 | 423.88 | 429.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 431.95 | 423.88 | 429.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 433.35 | 423.98 | 429.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:15:00 | 432.70 | 423.98 | 429.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 427.70 | 425.03 | 430.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 428.20 | 425.03 | 430.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 423.70 | 425.02 | 430.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 13:00:00 | 420.50 | 424.96 | 429.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 416.65 | 424.91 | 429.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 11:15:00 | 422.00 | 424.00 | 428.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:15:00 | 420.50 | 423.99 | 428.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 426.30 | 423.74 | 428.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 426.30 | 423.74 | 428.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 15:15:00 | 400.90 | 421.23 | 426.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 399.47 | 419.44 | 425.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 399.47 | 419.44 | 425.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | SL hit (close>ema200) qty=0.50 sl=419.19 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | SL hit (close>static) qty=1.00 sl=432.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | SL hit (close>ema200) qty=0.50 sl=419.19 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 435.75 | 419.19 | 425.28 | SL hit (close>ema200) qty=0.50 sl=419.19 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 435.75 | 419.19 | 425.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 437.30 | 419.37 | 425.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 429.45 | 419.37 | 425.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 407.98 | 420.03 | 425.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 421.65 | 420.04 | 425.11 | SL hit (close>ema200) qty=0.50 sl=420.04 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:45:00 | 433.20 | 410.46 | 415.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:30:00 | 433.85 | 411.61 | 416.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 14:15:00 | 442.50 | 412.70 | 416.51 | SL hit (close>static) qty=1.00 sl=441.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 14:15:00 | 442.50 | 412.70 | 416.51 | SL hit (close>static) qty=1.00 sl=441.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:00:00 | 433.00 | 413.16 | 416.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 441.20 | 415.49 | 417.72 | SL hit (close>static) qty=1.00 sl=441.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 448.80 | 419.91 | 419.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 450.55 | 421.35 | 420.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-19 13:30:00 | 328.80 | 2025-05-19 14:15:00 | 335.10 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-20 09:45:00 | 330.40 | 2025-05-20 10:15:00 | 336.40 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-05-21 09:15:00 | 327.15 | 2025-05-28 11:15:00 | 310.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 09:15:00 | 327.15 | 2025-06-09 09:15:00 | 313.20 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2025-05-28 11:30:00 | 311.70 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -6.66% |
| SELL | retest2 | 2025-06-09 15:00:00 | 315.55 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-06-10 09:15:00 | 314.30 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -5.77% |
| SELL | retest2 | 2025-06-23 09:15:00 | 311.80 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -6.62% |
| SELL | retest2 | 2025-06-23 10:30:00 | 309.50 | 2025-06-24 09:15:00 | 332.45 | STOP_HIT | 1.00 | -7.42% |
| BUY | retest2 | 2025-08-14 10:15:00 | 352.60 | 2025-08-14 12:15:00 | 344.45 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-08-14 14:45:00 | 354.55 | 2025-08-25 13:15:00 | 387.48 | TARGET_HIT | 1.00 | 9.29% |
| BUY | retest2 | 2025-08-18 11:15:00 | 352.25 | 2025-08-26 09:15:00 | 390.01 | TARGET_HIT | 1.00 | 10.72% |
| SELL | retest2 | 2026-02-11 13:00:00 | 420.50 | 2026-02-20 15:15:00 | 400.90 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2026-02-13 09:15:00 | 416.65 | 2026-02-24 12:15:00 | 399.47 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2026-02-16 11:15:00 | 422.00 | 2026-02-24 12:15:00 | 399.47 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2026-02-11 13:00:00 | 420.50 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 0.50 | -3.63% |
| SELL | retest2 | 2026-02-13 09:15:00 | 416.65 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 0.50 | -4.58% |
| SELL | retest2 | 2026-02-16 11:15:00 | 422.00 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 0.50 | -3.26% |
| SELL | retest2 | 2026-02-16 12:15:00 | 420.50 | 2026-02-25 09:15:00 | 435.75 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2026-02-25 11:15:00 | 429.45 | 2026-03-02 09:15:00 | 407.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 429.45 | 2026-03-02 10:15:00 | 421.65 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2026-04-09 12:45:00 | 433.20 | 2026-04-10 14:15:00 | 442.50 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-10 10:30:00 | 433.85 | 2026-04-10 14:15:00 | 442.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-04-13 10:00:00 | 433.00 | 2026-04-15 12:15:00 | 441.20 | STOP_HIT | 1.00 | -1.89% |
