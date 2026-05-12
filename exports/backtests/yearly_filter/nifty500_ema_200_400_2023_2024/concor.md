# Container Corporation of India Ltd. (CONCOR)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 533.75
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
| ALERT2_SKIP | 0 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 42 |
| PARTIAL | 13 |
| TARGET_HIT | 11 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 17
- **Target hits / Stop hits / Partials:** 11 / 31 / 13
- **Avg / median % per leg:** 2.99% / 3.32%
- **Sum % (uncompounded):** 164.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 12 | 46.2% | 7 | 19 | 0 | 1.46% | 37.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 12 | 46.2% | 7 | 19 | 0 | 1.46% | 37.9% |
| SELL (all) | 29 | 26 | 89.7% | 4 | 12 | 13 | 4.36% | 126.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 26 | 89.7% | 4 | 12 | 13 | 4.36% | 126.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 38 | 69.1% | 11 | 31 | 13 | 2.99% | 164.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 507.52 | 496.65 | 496.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 10:15:00 | 511.44 | 497.40 | 496.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 12:15:00 | 517.80 | 522.70 | 513.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-13 13:00:00 | 517.80 | 522.70 | 513.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 504.88 | 524.39 | 515.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:00:00 | 504.88 | 524.39 | 515.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 507.08 | 524.22 | 515.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 504.88 | 524.22 | 515.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 516.68 | 523.28 | 515.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 12:30:00 | 519.88 | 523.26 | 515.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 14:15:00 | 514.04 | 522.85 | 515.50 | SL hit (close<static) qty=1.00 sl=514.60 alert=retest2 |

### Cycle 2 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 801.04 | 825.92 | 825.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 797.92 | 825.64 | 825.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 690.00 | 689.51 | 719.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:45:00 | 689.36 | 689.51 | 719.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 683.88 | 664.85 | 686.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 685.84 | 664.85 | 686.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 688.04 | 665.08 | 686.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:30:00 | 690.04 | 665.08 | 686.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 691.44 | 665.34 | 686.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 691.44 | 665.34 | 686.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 686.16 | 665.99 | 686.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:45:00 | 686.44 | 665.99 | 686.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 688.44 | 666.21 | 686.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 685.96 | 666.21 | 686.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 683.88 | 666.38 | 686.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:00:00 | 682.72 | 666.95 | 686.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 683.40 | 667.11 | 686.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 15:00:00 | 682.64 | 667.27 | 686.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 669.20 | 668.55 | 686.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 13:15:00 | 648.58 | 665.78 | 682.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 13:15:00 | 649.23 | 665.78 | 682.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 13:15:00 | 648.51 | 665.78 | 682.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 635.74 | 663.94 | 681.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-20 14:15:00 | 614.45 | 659.41 | 677.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 585.24 | 561.70 | 561.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 589.92 | 561.98 | 561.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 604.40 | 604.91 | 588.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 602.56 | 604.91 | 588.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 589.36 | 604.56 | 590.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 590.44 | 604.56 | 590.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 585.72 | 604.37 | 589.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 585.56 | 604.37 | 589.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 588.28 | 603.06 | 589.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 588.28 | 603.06 | 589.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 586.24 | 602.90 | 589.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 584.92 | 602.90 | 589.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 588.00 | 602.47 | 589.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 587.60 | 602.47 | 589.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 589.00 | 602.33 | 589.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:30:00 | 591.36 | 601.85 | 589.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 595.60 | 601.58 | 589.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 590.70 | 600.34 | 592.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:30:00 | 591.60 | 600.31 | 592.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 594.00 | 600.24 | 592.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 594.25 | 600.24 | 592.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 598.70 | 600.23 | 592.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 600.25 | 600.21 | 592.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 604.00 | 600.29 | 592.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 600.85 | 608.09 | 600.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:30:00 | 600.10 | 607.92 | 600.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 600.05 | 607.84 | 600.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 598.45 | 607.84 | 600.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 599.70 | 607.76 | 600.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:45:00 | 598.55 | 607.76 | 600.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 598.10 | 607.66 | 600.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 600.20 | 607.66 | 600.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 597.55 | 607.49 | 600.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 597.55 | 607.49 | 600.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 593.70 | 607.35 | 600.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 593.70 | 607.35 | 600.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 591.30 | 607.19 | 600.25 | SL hit (close<static) qty=1.00 sl=592.65 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 537.50 | 595.03 | 595.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 534.00 | 594.42 | 594.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 561.90 | 553.67 | 565.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 561.90 | 553.67 | 565.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 566.45 | 554.00 | 565.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 568.60 | 554.00 | 565.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 565.25 | 554.11 | 565.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 563.85 | 554.21 | 565.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 564.05 | 554.30 | 565.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:30:00 | 564.05 | 554.48 | 565.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 562.50 | 554.66 | 565.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 535.66 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 535.85 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 535.85 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 534.38 | 554.37 | 563.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 544.10 | 541.61 | 552.95 | SL hit (close>ema200) qty=0.50 sl=541.61 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 518.40 | 486.83 | 486.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 521.20 | 493.03 | 490.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-21 12:30:00 | 519.88 | 2023-06-22 14:15:00 | 514.04 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-06-23 12:30:00 | 519.84 | 2023-06-23 14:15:00 | 514.48 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-06-26 11:30:00 | 518.08 | 2023-08-08 15:15:00 | 569.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-26 12:15:00 | 518.20 | 2023-08-08 15:15:00 | 570.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-30 11:00:00 | 545.16 | 2023-10-25 12:15:00 | 553.68 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2023-08-31 11:00:00 | 545.60 | 2023-10-25 12:15:00 | 553.68 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2023-09-01 09:15:00 | 550.52 | 2023-10-25 12:15:00 | 553.68 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2023-09-01 12:45:00 | 547.04 | 2023-10-25 12:15:00 | 553.68 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2023-10-12 12:45:00 | 562.40 | 2023-10-25 12:15:00 | 553.68 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2023-10-12 15:00:00 | 561.80 | 2023-10-25 12:15:00 | 553.68 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-10-13 09:30:00 | 561.28 | 2023-10-25 12:15:00 | 553.68 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-10-23 14:00:00 | 564.68 | 2023-10-27 12:15:00 | 553.28 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2023-10-25 09:15:00 | 562.92 | 2023-10-30 13:15:00 | 551.48 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-10-25 11:45:00 | 563.36 | 2023-11-15 10:15:00 | 599.68 | TARGET_HIT | 1.00 | 6.45% |
| BUY | retest2 | 2023-10-25 12:15:00 | 561.52 | 2023-11-15 10:15:00 | 600.16 | TARGET_HIT | 1.00 | 6.88% |
| BUY | retest2 | 2023-10-27 12:15:00 | 561.72 | 2023-11-15 10:15:00 | 605.57 | TARGET_HIT | 1.00 | 7.81% |
| BUY | retest2 | 2023-10-30 10:15:00 | 556.48 | 2023-11-15 10:15:00 | 601.74 | TARGET_HIT | 1.00 | 8.13% |
| BUY | retest2 | 2023-11-03 09:15:00 | 571.08 | 2023-12-01 14:15:00 | 628.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-10 13:00:00 | 682.72 | 2024-12-17 13:15:00 | 648.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 14:15:00 | 683.40 | 2024-12-17 13:15:00 | 649.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 15:00:00 | 682.64 | 2024-12-17 13:15:00 | 648.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 669.20 | 2024-12-18 14:15:00 | 635.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 13:00:00 | 682.72 | 2024-12-20 14:15:00 | 614.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-10 14:15:00 | 683.40 | 2024-12-20 14:15:00 | 615.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-10 15:00:00 | 682.64 | 2024-12-20 14:15:00 | 614.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 669.20 | 2025-01-07 11:15:00 | 602.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 15:15:00 | 566.40 | 2025-04-07 09:15:00 | 538.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 12:45:00 | 567.88 | 2025-04-07 09:15:00 | 539.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 563.16 | 2025-04-07 09:15:00 | 535.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 15:15:00 | 566.40 | 2025-04-09 14:15:00 | 549.00 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2025-04-03 12:45:00 | 567.88 | 2025-04-09 14:15:00 | 549.00 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-04-04 09:15:00 | 563.16 | 2025-04-09 14:15:00 | 549.00 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-04-21 15:15:00 | 569.00 | 2025-04-25 11:15:00 | 540.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-21 15:15:00 | 569.00 | 2025-04-29 09:15:00 | 554.72 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-05-14 14:30:00 | 550.32 | 2025-05-16 09:15:00 | 561.60 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-05-14 15:15:00 | 550.20 | 2025-05-16 09:15:00 | 561.60 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-06-23 13:30:00 | 591.36 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-06-24 09:15:00 | 595.60 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-08 11:45:00 | 590.70 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-07-08 12:30:00 | 591.60 | 2025-07-28 12:15:00 | 591.30 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-07-09 09:15:00 | 600.25 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-07-09 09:45:00 | 604.00 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2025-07-25 11:15:00 | 600.85 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-07-25 12:30:00 | 600.10 | 2025-07-31 09:15:00 | 578.80 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-09-16 12:00:00 | 563.85 | 2025-09-24 10:15:00 | 535.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:30:00 | 564.05 | 2025-09-24 10:15:00 | 535.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 14:30:00 | 564.05 | 2025-09-24 10:15:00 | 535.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 562.50 | 2025-09-24 10:15:00 | 534.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:00:00 | 563.85 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-09-16 12:30:00 | 564.05 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-09-16 14:30:00 | 564.05 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-09-17 09:45:00 | 562.50 | 2025-10-10 10:15:00 | 544.10 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-10-31 15:00:00 | 544.25 | 2025-11-03 09:15:00 | 549.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-04 11:00:00 | 544.65 | 2025-11-07 09:15:00 | 517.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:00:00 | 544.65 | 2025-11-12 11:15:00 | 537.50 | STOP_HIT | 0.50 | 1.31% |
