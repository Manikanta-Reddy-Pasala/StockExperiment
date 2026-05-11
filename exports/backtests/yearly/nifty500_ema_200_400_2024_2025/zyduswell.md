# Zydus Wellness Ltd. (ZYDUSWELL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 517.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 23 |
| PARTIAL | 11 |
| TARGET_HIT | 11 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 14
- **Target hits / Stop hits / Partials:** 11 / 16 / 11
- **Avg / median % per leg:** 2.91% / 5.00%
- **Sum % (uncompounded):** 110.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 35 | 21 | 60.0% | 8 | 16 | 11 | 2.31% | 80.8% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -7.99% | -32.0% |
| SELL @ 3rd Alert (retest2) | 31 | 21 | 67.7% | 8 | 12 | 11 | 3.64% | 112.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -7.99% | -32.0% |
| retest2 (combined) | 34 | 24 | 70.6% | 11 | 12 | 11 | 4.20% | 142.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 389.70 | 426.76 | 426.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 387.98 | 424.98 | 425.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 10:15:00 | 391.18 | 391.03 | 403.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:00:00 | 391.18 | 391.03 | 403.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 401.98 | 391.17 | 401.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 397.01 | 391.17 | 401.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 397.43 | 391.23 | 401.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:00:00 | 394.14 | 391.41 | 401.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 379.75 | 391.42 | 401.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 374.43 | 390.49 | 400.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 389.51 | 389.40 | 399.22 | SL hit (close>ema200) qty=0.50 sl=389.40 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 359.22 | 347.19 | 347.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 362.60 | 347.73 | 347.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 405.34 | 406.24 | 394.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 405.34 | 406.24 | 394.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 402.80 | 405.89 | 395.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:45:00 | 394.84 | 405.89 | 395.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 398.14 | 406.63 | 396.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 398.14 | 406.63 | 396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 397.14 | 406.44 | 396.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 397.30 | 406.44 | 396.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 396.52 | 406.34 | 396.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 396.52 | 406.34 | 396.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 396.20 | 406.24 | 396.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 393.64 | 406.14 | 396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 393.98 | 406.02 | 396.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 393.60 | 406.02 | 396.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 394.28 | 405.90 | 396.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:45:00 | 395.48 | 399.56 | 395.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 394.98 | 399.41 | 395.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:45:00 | 397.34 | 399.39 | 395.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-01 10:15:00 | 435.03 | 400.82 | 397.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 430.25 | 454.54 | 454.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 429.30 | 453.82 | 454.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 434.45 | 433.76 | 441.16 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:15:00 | 426.45 | 433.71 | 441.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 13:00:00 | 426.85 | 433.08 | 440.42 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 15:00:00 | 426.20 | 432.95 | 440.28 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:15:00 | 424.70 | 432.90 | 440.22 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 460.10 | 431.65 | 438.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 460.10 | 431.65 | 438.79 | SL hit (close>ema400) qty=1.00 sl=438.79 alert=retest1 |

### Cycle 4 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 475.30 | 444.75 | 444.63 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 437.30 | 444.62 | 444.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 432.95 | 444.37 | 444.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 437.00 | 435.68 | 439.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 10:15:00 | 437.00 | 435.68 | 439.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 437.00 | 435.68 | 439.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 437.00 | 435.68 | 439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 441.80 | 435.74 | 439.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 441.80 | 435.74 | 439.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 440.20 | 435.79 | 439.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 440.20 | 435.79 | 439.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 442.35 | 435.85 | 439.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:45:00 | 442.95 | 435.85 | 439.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 447.50 | 435.97 | 439.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 447.50 | 435.97 | 439.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 439.50 | 436.47 | 439.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 428.45 | 436.54 | 439.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:15:00 | 407.03 | 434.86 | 438.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-26 11:15:00 | 385.61 | 412.23 | 422.68 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 522.80 | 419.18 | 418.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 09:15:00 | 531.55 | 439.94 | 430.02 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-08 14:00:00 | 394.14 | 2024-11-13 09:15:00 | 374.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:00:00 | 394.14 | 2024-11-14 12:15:00 | 389.51 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2024-11-11 12:15:00 | 379.75 | 2024-11-28 11:15:00 | 406.18 | STOP_HIT | 1.00 | -6.96% |
| SELL | retest2 | 2024-11-27 11:15:00 | 394.36 | 2024-11-28 11:15:00 | 406.18 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-12-19 09:45:00 | 394.20 | 2024-12-30 14:15:00 | 374.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:45:00 | 394.20 | 2024-12-31 14:15:00 | 396.80 | STOP_HIT | 0.50 | -0.66% |
| SELL | retest2 | 2025-01-02 09:45:00 | 394.08 | 2025-01-02 14:15:00 | 402.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-01-06 09:15:00 | 391.83 | 2025-01-13 09:15:00 | 374.38 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2025-01-07 14:30:00 | 394.08 | 2025-01-13 11:15:00 | 372.24 | PARTIAL | 0.50 | 5.54% |
| SELL | retest2 | 2025-01-07 15:00:00 | 393.54 | 2025-01-13 11:15:00 | 373.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 09:15:00 | 391.83 | 2025-01-27 09:15:00 | 352.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-07 14:30:00 | 394.08 | 2025-01-27 09:15:00 | 354.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-07 15:00:00 | 393.54 | 2025-01-27 09:15:00 | 354.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 348.70 | 2025-02-14 10:15:00 | 331.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 11:45:00 | 348.40 | 2025-02-14 10:15:00 | 330.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 13:00:00 | 349.01 | 2025-02-14 10:15:00 | 331.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 14:00:00 | 349.95 | 2025-02-14 10:15:00 | 332.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 348.70 | 2025-02-27 12:15:00 | 314.95 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2025-02-11 11:45:00 | 348.40 | 2025-02-28 09:15:00 | 313.83 | TARGET_HIT | 0.50 | 9.92% |
| SELL | retest2 | 2025-02-11 13:00:00 | 349.01 | 2025-02-28 09:15:00 | 313.56 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2025-02-11 14:00:00 | 349.95 | 2025-02-28 09:15:00 | 314.11 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2025-03-28 11:15:00 | 339.45 | 2025-04-08 13:15:00 | 350.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-04-02 13:00:00 | 340.75 | 2025-04-08 13:15:00 | 350.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-04-03 11:00:00 | 340.30 | 2025-04-08 13:15:00 | 350.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-05-06 13:00:00 | 340.60 | 2025-05-12 09:15:00 | 354.60 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-08-18 09:45:00 | 395.48 | 2025-09-01 10:15:00 | 435.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 11:30:00 | 394.98 | 2025-09-01 10:15:00 | 434.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 12:45:00 | 397.34 | 2025-09-01 10:15:00 | 437.07 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-12-23 10:15:00 | 426.45 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -7.89% |
| SELL | retest1 | 2025-12-24 13:00:00 | 426.85 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -7.79% |
| SELL | retest1 | 2025-12-24 15:00:00 | 426.20 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -7.95% |
| SELL | retest1 | 2025-12-26 09:15:00 | 424.70 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -8.34% |
| SELL | retest2 | 2026-02-02 09:15:00 | 428.45 | 2026-02-04 10:15:00 | 407.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-02 09:15:00 | 428.45 | 2026-02-26 11:15:00 | 385.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-13 13:30:00 | 431.50 | 2026-03-13 14:15:00 | 409.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 13:30:00 | 431.50 | 2026-03-13 14:15:00 | 401.20 | STOP_HIT | 0.50 | 7.02% |
| SELL | retest2 | 2026-03-24 13:30:00 | 436.35 | 2026-03-25 09:15:00 | 446.10 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-03-27 14:15:00 | 435.80 | 2026-03-27 14:15:00 | 448.05 | STOP_HIT | 1.00 | -2.81% |
