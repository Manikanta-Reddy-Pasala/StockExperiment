# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 598.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 67 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 53 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 37
- **Target hits / Stop hits / Partials:** 9 / 44 / 9
- **Avg / median % per leg:** 0.83% / -1.37%
- **Sum % (uncompounded):** 51.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 7 | 10 | 0 | 2.27% | 38.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 7 | 41.2% | 7 | 10 | 0 | 2.27% | 38.6% |
| SELL (all) | 45 | 18 | 40.0% | 2 | 34 | 9 | 0.29% | 13.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 18 | 40.0% | 2 | 34 | 9 | 0.29% | 13.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 25 | 40.3% | 9 | 44 | 9 | 0.83% | 51.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 371.25 | 399.38 | 399.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 362.55 | 395.23 | 397.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 13:15:00 | 378.55 | 374.70 | 385.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 14:00:00 | 378.55 | 374.70 | 385.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 391.30 | 374.87 | 385.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:00:00 | 391.30 | 374.87 | 385.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 392.55 | 375.04 | 385.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 382.65 | 375.04 | 385.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 11:15:00 | 395.95 | 375.57 | 385.23 | SL hit (close>static) qty=1.00 sl=393.75 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 14:15:00 | 420.55 | 390.99 | 390.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 433.30 | 399.22 | 395.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 09:15:00 | 402.80 | 408.54 | 401.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 10:00:00 | 402.80 | 408.54 | 401.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 399.95 | 408.45 | 401.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:00:00 | 399.95 | 408.45 | 401.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 11:15:00 | 400.70 | 408.38 | 401.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:45:00 | 399.15 | 408.38 | 401.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 406.45 | 408.09 | 401.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 09:15:00 | 417.00 | 407.89 | 401.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 12:00:00 | 412.40 | 408.10 | 401.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 12:45:00 | 414.70 | 408.16 | 402.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 15:15:00 | 399.00 | 408.17 | 402.91 | SL hit (close<static) qty=1.00 sl=399.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 10:15:00 | 377.50 | 404.07 | 404.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 14:15:00 | 374.05 | 402.94 | 403.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 395.90 | 393.76 | 398.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 395.90 | 393.76 | 398.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 395.90 | 393.76 | 398.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:00:00 | 395.90 | 393.76 | 398.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 399.15 | 393.81 | 398.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:45:00 | 402.30 | 393.81 | 398.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 11:15:00 | 411.90 | 393.99 | 398.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 11:45:00 | 409.75 | 393.99 | 398.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 419.30 | 394.24 | 398.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 13:00:00 | 419.30 | 394.24 | 398.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 14:15:00 | 469.60 | 403.21 | 402.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 09:15:00 | 488.65 | 432.95 | 421.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 451.60 | 456.65 | 438.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 10:00:00 | 451.60 | 456.65 | 438.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 537.70 | 569.99 | 544.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 537.70 | 569.99 | 544.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 556.40 | 569.85 | 544.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 09:45:00 | 560.00 | 569.13 | 544.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 560.30 | 568.34 | 546.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 13:30:00 | 558.30 | 567.73 | 546.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:45:00 | 558.75 | 567.63 | 546.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-26 10:15:00 | 616.00 | 575.91 | 557.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 634.00 | 706.07 | 706.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 623.15 | 703.82 | 705.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 11:15:00 | 691.25 | 687.76 | 696.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 12:00:00 | 691.25 | 687.76 | 696.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 693.25 | 687.82 | 696.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 693.25 | 687.82 | 696.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 686.90 | 687.81 | 696.09 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 853.75 | 703.53 | 703.00 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 684.50 | 734.70 | 734.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 682.55 | 733.69 | 734.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 10:15:00 | 579.10 | 570.96 | 618.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-07 11:00:00 | 579.10 | 570.96 | 618.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 610.40 | 572.24 | 618.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 589.45 | 572.24 | 618.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 15:15:00 | 559.98 | 572.40 | 616.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-11 09:15:00 | 530.51 | 571.98 | 616.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 585.00 | 562.00 | 561.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 602.40 | 572.07 | 567.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 632.75 | 632.84 | 612.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:00:00 | 632.75 | 632.84 | 612.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 613.25 | 632.09 | 613.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 612.45 | 632.09 | 613.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 611.15 | 631.88 | 613.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 610.75 | 631.88 | 613.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 608.80 | 631.65 | 613.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 609.80 | 631.65 | 613.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 615.05 | 631.10 | 613.16 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 528.90 | 602.35 | 602.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 516.30 | 597.94 | 600.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 537.30 | 536.57 | 557.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 537.30 | 536.57 | 557.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 535.00 | 526.17 | 536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 535.00 | 526.17 | 536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 535.85 | 526.26 | 536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 535.70 | 526.26 | 536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 537.95 | 526.38 | 536.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 539.55 | 526.38 | 536.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 532.90 | 526.44 | 536.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 529.10 | 526.69 | 536.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:45:00 | 529.15 | 528.62 | 536.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 553.50 | 527.58 | 534.96 | SL hit (close>static) qty=1.00 sl=543.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 570.25 | 486.80 | 486.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 577.90 | 488.53 | 487.50 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-03 09:15:00 | 382.65 | 2023-11-03 11:15:00 | 395.95 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2023-11-03 13:45:00 | 389.80 | 2023-11-09 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2023-11-06 09:15:00 | 389.95 | 2023-11-09 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2023-11-08 09:45:00 | 387.85 | 2023-11-09 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2023-12-14 09:15:00 | 417.00 | 2023-12-20 15:15:00 | 399.00 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2023-12-14 12:00:00 | 412.40 | 2023-12-20 15:15:00 | 399.00 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2023-12-14 12:45:00 | 414.70 | 2023-12-20 15:15:00 | 399.00 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2023-12-27 14:45:00 | 415.10 | 2024-01-09 09:15:00 | 404.40 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-01-09 09:15:00 | 407.35 | 2024-01-10 10:15:00 | 396.95 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-01-15 09:30:00 | 407.40 | 2024-01-18 09:15:00 | 396.60 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-01-15 10:00:00 | 407.70 | 2024-01-18 09:15:00 | 396.60 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-01-17 09:30:00 | 407.10 | 2024-01-18 09:15:00 | 396.60 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-06-05 09:45:00 | 560.00 | 2024-06-26 10:15:00 | 616.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 09:15:00 | 560.30 | 2024-06-26 10:15:00 | 616.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 13:30:00 | 558.30 | 2024-06-26 10:15:00 | 614.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 14:45:00 | 558.75 | 2024-06-26 10:15:00 | 614.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-24 14:30:00 | 612.20 | 2024-08-01 10:15:00 | 586.55 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-07-30 15:15:00 | 603.40 | 2024-08-01 10:15:00 | 586.55 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-08-01 14:30:00 | 605.00 | 2024-08-07 09:15:00 | 665.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-02 10:45:00 | 604.55 | 2024-08-07 09:15:00 | 665.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 617.70 | 2024-08-08 10:15:00 | 679.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-10 09:15:00 | 589.45 | 2025-03-10 15:15:00 | 559.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 09:15:00 | 589.45 | 2025-03-11 09:15:00 | 530.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-15 10:15:00 | 602.05 | 2025-05-21 09:15:00 | 571.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-15 10:15:00 | 602.05 | 2025-05-21 09:15:00 | 573.30 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2025-05-16 09:45:00 | 602.70 | 2025-05-21 09:15:00 | 572.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-16 09:45:00 | 602.70 | 2025-05-21 09:15:00 | 573.30 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2025-05-19 09:30:00 | 603.10 | 2025-05-21 09:15:00 | 572.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 09:30:00 | 603.10 | 2025-05-21 09:15:00 | 573.30 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-10-30 13:30:00 | 529.10 | 2025-11-11 09:15:00 | 553.50 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-11-06 09:45:00 | 529.15 | 2025-11-11 09:15:00 | 553.50 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-11-12 09:15:00 | 528.40 | 2025-11-12 11:15:00 | 543.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-12-01 09:45:00 | 528.35 | 2025-12-03 14:15:00 | 546.50 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-12-10 14:15:00 | 533.70 | 2025-12-15 15:15:00 | 539.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-16 09:15:00 | 529.40 | 2025-12-23 09:15:00 | 543.15 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-12-22 11:45:00 | 535.55 | 2025-12-23 09:15:00 | 543.15 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-12-22 12:15:00 | 535.80 | 2025-12-23 09:15:00 | 543.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-12-23 11:15:00 | 537.90 | 2025-12-24 10:15:00 | 545.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-23 12:45:00 | 537.30 | 2025-12-24 10:15:00 | 545.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-23 14:15:00 | 537.95 | 2025-12-24 10:15:00 | 545.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-12-30 09:45:00 | 535.80 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-31 13:00:00 | 537.00 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-01 09:15:00 | 537.05 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-05 09:15:00 | 536.10 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-05 11:15:00 | 537.25 | 2026-01-12 09:15:00 | 509.01 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2026-01-05 11:15:00 | 537.25 | 2026-01-19 13:15:00 | 482.22 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2026-02-04 11:45:00 | 511.95 | 2026-02-11 11:15:00 | 486.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 13:00:00 | 511.50 | 2026-02-11 11:15:00 | 485.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 13:45:00 | 511.00 | 2026-02-11 11:15:00 | 485.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 09:45:00 | 508.00 | 2026-02-12 09:15:00 | 482.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 11:45:00 | 511.95 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-02-09 13:00:00 | 511.50 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2026-02-09 13:45:00 | 511.00 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-02-10 09:45:00 | 508.00 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-04-09 13:00:00 | 457.60 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.38% |
| SELL | retest2 | 2026-04-10 14:15:00 | 458.30 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.22% |
| SELL | retest2 | 2026-04-10 15:15:00 | 459.10 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.03% |
| SELL | retest2 | 2026-04-13 12:00:00 | 458.85 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2026-04-15 12:30:00 | 473.05 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-16 11:45:00 | 474.15 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-04-16 13:15:00 | 475.45 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-04-16 14:15:00 | 475.00 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.11% |
