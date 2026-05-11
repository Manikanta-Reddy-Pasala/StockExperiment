# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 742.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 24 |
| PARTIAL | 1 |
| TARGET_HIT | 15 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 10
- **Target hits / Stop hits / Partials:** 15 / 10 / 1
- **Avg / median % per leg:** 4.71% / 10.00%
- **Sum % (uncompounded):** 122.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 14 | 70.0% | 14 | 6 | 0 | 6.18% | 123.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 14 | 70.0% | 14 | 6 | 0 | 6.18% | 123.6% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.21% | -1.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.06% | -16.2% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 24 | 14 | 58.3% | 14 | 10 | 0 | 4.47% | 107.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 12:15:00 | 350.35 | 436.34 | 436.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 349.25 | 434.63 | 435.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 379.55 | 377.26 | 397.24 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 09:15:00 | 371.05 | 377.42 | 396.73 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 09:15:00 | 352.50 | 377.37 | 396.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-06-04 11:15:00 | 333.94 | 371.83 | 390.99 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 10:15:00 | 388.20 | 362.95 | 362.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 390.05 | 364.90 | 363.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 407.40 | 409.49 | 397.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 11:00:00 | 407.40 | 409.49 | 397.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 405.15 | 415.57 | 405.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 14:30:00 | 404.95 | 415.57 | 405.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 404.00 | 415.45 | 405.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 403.00 | 415.45 | 405.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 405.55 | 415.35 | 405.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 402.80 | 415.35 | 405.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 407.70 | 415.28 | 405.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 450.15 | 414.76 | 405.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-29 09:15:00 | 495.17 | 438.25 | 426.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 431.90 | 476.14 | 476.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 422.40 | 475.61 | 476.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 12:15:00 | 437.45 | 432.10 | 447.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 13:00:00 | 437.45 | 432.10 | 447.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 440.00 | 432.00 | 443.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:45:00 | 446.30 | 432.00 | 443.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 443.40 | 432.47 | 443.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 443.40 | 432.47 | 443.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 443.55 | 432.58 | 443.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 437.90 | 432.58 | 443.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 439.35 | 432.65 | 443.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:00:00 | 436.80 | 432.69 | 443.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:00:00 | 437.00 | 432.80 | 443.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 15:15:00 | 436.35 | 432.96 | 443.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:45:00 | 435.20 | 433.01 | 443.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 447.45 | 433.28 | 443.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 447.45 | 433.28 | 443.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 454.05 | 433.49 | 443.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 454.05 | 433.49 | 443.30 | SL hit (close>static) qty=1.00 sl=448.10 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 481.50 | 451.14 | 451.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 485.00 | 451.48 | 451.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 12:15:00 | 557.90 | 558.68 | 536.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 13:00:00 | 557.90 | 558.68 | 536.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 576.55 | 591.54 | 573.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 582.05 | 591.54 | 573.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 580.30 | 592.38 | 578.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:00:00 | 578.75 | 592.14 | 578.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-02 15:15:00 | 640.25 | 602.01 | 590.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 10:15:00 | 621.85 | 656.02 | 656.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 610.30 | 653.87 | 654.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 625.35 | 623.60 | 635.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:30:00 | 625.00 | 623.60 | 635.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 606.80 | 575.96 | 596.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 606.80 | 575.96 | 596.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 608.50 | 576.29 | 596.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 608.50 | 576.29 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 600.00 | 577.55 | 596.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 604.95 | 577.55 | 596.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 597.10 | 579.12 | 597.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:30:00 | 599.30 | 579.12 | 597.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 599.55 | 579.32 | 597.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:45:00 | 599.70 | 579.32 | 597.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 600.05 | 579.53 | 597.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 599.95 | 579.53 | 597.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 614.85 | 580.78 | 597.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 614.85 | 580.78 | 597.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 654.70 | 608.75 | 608.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 11:15:00 | 662.90 | 615.07 | 611.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 639.45 | 641.15 | 628.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 11:00:00 | 639.45 | 641.15 | 628.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 625.40 | 640.64 | 628.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 625.40 | 640.64 | 628.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 627.85 | 640.51 | 628.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 627.85 | 640.51 | 628.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 632.00 | 640.43 | 628.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 617.65 | 640.43 | 628.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 614.40 | 640.17 | 628.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:45:00 | 612.95 | 640.17 | 628.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 632.00 | 638.59 | 628.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:30:00 | 624.80 | 638.59 | 628.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 641.60 | 638.53 | 628.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 644.95 | 638.53 | 628.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-30 15:15:00 | 709.45 | 641.79 | 631.11 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-16 10:45:00 | 305.15 | 2023-08-28 09:15:00 | 335.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-18 09:15:00 | 305.40 | 2023-08-28 09:15:00 | 335.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-18 10:15:00 | 305.35 | 2023-08-28 09:15:00 | 335.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-18 10:45:00 | 305.75 | 2023-08-28 09:15:00 | 336.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-10 11:45:00 | 325.80 | 2023-10-25 14:15:00 | 358.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-10 12:30:00 | 325.35 | 2023-10-25 14:15:00 | 357.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-10 15:15:00 | 326.70 | 2023-10-25 14:15:00 | 359.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-11 10:00:00 | 325.15 | 2023-10-25 14:15:00 | 357.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-15 12:45:00 | 431.50 | 2024-03-19 09:15:00 | 428.75 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-03-15 14:00:00 | 431.20 | 2024-03-19 09:15:00 | 428.75 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-03-18 10:15:00 | 431.80 | 2024-03-19 09:15:00 | 428.75 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-03-18 12:00:00 | 431.40 | 2024-03-19 09:15:00 | 428.75 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-03-20 14:30:00 | 446.50 | 2024-03-27 09:15:00 | 410.70 | STOP_HIT | 1.00 | -8.02% |
| BUY | retest2 | 2024-03-26 09:45:00 | 436.20 | 2024-03-27 09:15:00 | 410.70 | STOP_HIT | 1.00 | -5.85% |
| BUY | retest2 | 2024-04-04 11:00:00 | 434.70 | 2024-04-08 09:15:00 | 478.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-05-29 09:15:00 | 371.05 | 2024-05-29 09:15:00 | 352.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-05-29 09:15:00 | 371.05 | 2024-06-04 11:15:00 | 333.94 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-24 09:15:00 | 450.15 | 2024-11-29 09:15:00 | 495.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-26 11:00:00 | 436.80 | 2025-03-27 15:15:00 | 454.05 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-03-26 13:00:00 | 437.00 | 2025-03-27 15:15:00 | 454.05 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-03-26 15:15:00 | 436.35 | 2025-03-27 15:15:00 | 454.05 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2025-03-27 09:45:00 | 435.20 | 2025-03-27 15:15:00 | 454.05 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-07-29 09:15:00 | 582.05 | 2025-09-02 15:15:00 | 640.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-08 15:00:00 | 580.30 | 2025-09-02 15:15:00 | 638.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 10:00:00 | 578.75 | 2025-09-02 15:15:00 | 636.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-25 10:15:00 | 644.95 | 2026-03-30 15:15:00 | 709.45 | TARGET_HIT | 1.00 | 10.00% |
