# Ambuja Cements Ltd. (AMBUJACEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 443.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 47 |
| PARTIAL | 16 |
| TARGET_HIT | 21 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 23
- **Target hits / Stop hits / Partials:** 21 / 26 / 16
- **Avg / median % per leg:** 4.00% / 5.00%
- **Sum % (uncompounded):** 252.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 8 | 38.1% | 7 | 14 | 0 | 2.10% | 44.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 8 | 38.1% | 7 | 14 | 0 | 2.10% | 44.2% |
| SELL (all) | 42 | 32 | 76.2% | 14 | 12 | 16 | 4.95% | 208.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 32 | 76.2% | 14 | 12 | 16 | 4.95% | 208.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 63 | 40 | 63.5% | 21 | 26 | 16 | 4.00% | 252.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 633.95 | 648.52 | 648.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 12:15:00 | 624.45 | 645.57 | 647.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 13:15:00 | 636.00 | 635.50 | 640.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 13:30:00 | 635.35 | 635.50 | 640.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 631.80 | 625.73 | 632.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 631.80 | 625.73 | 632.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 631.70 | 625.79 | 632.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 13:15:00 | 629.85 | 625.91 | 632.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 635.15 | 626.08 | 632.76 | SL hit (close>static) qty=1.00 sl=633.75 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 545.05 | 516.61 | 516.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 13:15:00 | 548.85 | 516.94 | 516.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 535.85 | 540.81 | 531.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 14:15:00 | 535.85 | 540.81 | 531.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 535.85 | 540.81 | 531.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 535.85 | 540.81 | 531.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 530.85 | 540.61 | 531.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:00:00 | 530.85 | 540.61 | 531.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 530.05 | 540.50 | 531.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:45:00 | 530.05 | 540.50 | 531.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 531.65 | 540.21 | 531.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:30:00 | 528.60 | 540.21 | 531.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 531.55 | 540.12 | 531.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 535.85 | 540.12 | 531.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 539.50 | 540.12 | 531.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 10:15:00 | 540.90 | 540.12 | 531.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 528.55 | 539.92 | 531.97 | SL hit (close<static) qty=1.00 sl=529.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 552.25 | 577.87 | 577.92 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 590.50 | 577.72 | 577.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 565.00 | 577.78 | 577.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 560.00 | 575.73 | 576.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 570.30 | 568.28 | 571.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:45:00 | 570.00 | 568.28 | 571.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 573.00 | 568.33 | 571.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 569.45 | 568.34 | 571.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 569.50 | 568.35 | 571.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 13:00:00 | 569.30 | 568.20 | 571.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 569.25 | 568.16 | 571.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 570.25 | 568.18 | 571.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 570.15 | 568.18 | 571.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 575.70 | 568.26 | 571.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 12:15:00 | 575.70 | 568.26 | 571.50 | SL hit (close>static) qty=1.00 sl=575.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 11:30:00 | 572.60 | 2024-06-05 09:15:00 | 554.15 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2024-06-04 13:15:00 | 581.55 | 2024-06-05 09:15:00 | 554.15 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2024-06-05 10:45:00 | 580.05 | 2024-06-10 09:15:00 | 638.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 11:30:00 | 620.30 | 2024-06-14 09:15:00 | 682.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 13:30:00 | 620.00 | 2024-06-14 09:15:00 | 682.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 14:30:00 | 620.60 | 2024-06-14 09:15:00 | 682.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 11:45:00 | 619.90 | 2024-08-22 14:15:00 | 633.95 | STOP_HIT | 1.00 | 2.27% |
| SELL | retest2 | 2024-09-27 13:15:00 | 629.85 | 2024-09-27 14:15:00 | 635.15 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-09-30 15:15:00 | 630.00 | 2024-10-01 12:15:00 | 634.05 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-10-01 09:30:00 | 630.00 | 2024-10-01 12:15:00 | 634.05 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-10-03 09:15:00 | 630.35 | 2024-10-07 10:15:00 | 598.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 630.35 | 2024-10-18 09:15:00 | 567.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-04 10:15:00 | 562.15 | 2024-12-09 11:15:00 | 572.90 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-12-19 09:15:00 | 555.45 | 2024-12-31 12:15:00 | 533.85 | PARTIAL | 0.50 | 3.89% |
| SELL | retest2 | 2024-12-20 09:15:00 | 561.95 | 2024-12-31 12:15:00 | 533.90 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2024-12-20 09:45:00 | 562.00 | 2025-01-06 10:15:00 | 527.68 | PARTIAL | 0.50 | 6.11% |
| SELL | retest2 | 2024-12-19 09:15:00 | 555.45 | 2025-01-13 09:15:00 | 505.76 | TARGET_HIT | 0.50 | 8.95% |
| SELL | retest2 | 2024-12-20 09:15:00 | 561.95 | 2025-01-13 09:15:00 | 505.80 | TARGET_HIT | 0.50 | 9.99% |
| SELL | retest2 | 2024-12-20 09:45:00 | 562.00 | 2025-01-13 14:15:00 | 499.91 | TARGET_HIT | 0.50 | 11.05% |
| SELL | retest2 | 2025-01-27 09:15:00 | 544.55 | 2025-01-29 15:15:00 | 519.60 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2025-01-28 13:15:00 | 546.95 | 2025-01-29 15:15:00 | 520.70 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-01-29 11:45:00 | 548.10 | 2025-01-30 09:15:00 | 517.32 | PARTIAL | 0.50 | 5.62% |
| SELL | retest2 | 2025-01-27 09:15:00 | 544.55 | 2025-02-01 12:15:00 | 490.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-28 13:15:00 | 546.95 | 2025-02-01 12:15:00 | 492.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-29 11:45:00 | 548.10 | 2025-02-01 12:15:00 | 493.29 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-05 10:15:00 | 540.90 | 2025-05-07 09:15:00 | 528.55 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-05-12 09:45:00 | 542.05 | 2025-06-19 12:15:00 | 533.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-05-12 13:15:00 | 540.35 | 2025-06-19 12:15:00 | 533.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-05-12 13:45:00 | 540.50 | 2025-06-19 12:15:00 | 533.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-06-13 10:30:00 | 545.70 | 2025-06-19 12:15:00 | 533.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-06-13 12:45:00 | 544.35 | 2025-07-02 13:15:00 | 596.25 | TARGET_HIT | 1.00 | 9.54% |
| BUY | retest2 | 2025-06-13 15:15:00 | 543.95 | 2025-07-02 13:15:00 | 594.39 | TARGET_HIT | 1.00 | 9.27% |
| BUY | retest2 | 2025-06-16 12:00:00 | 544.95 | 2025-07-02 13:15:00 | 594.55 | TARGET_HIT | 1.00 | 9.10% |
| BUY | retest2 | 2025-08-07 14:00:00 | 586.30 | 2025-08-08 14:15:00 | 580.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-08 13:30:00 | 585.80 | 2025-08-08 14:15:00 | 580.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-11 10:15:00 | 586.25 | 2025-08-13 10:15:00 | 578.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-18 09:15:00 | 596.00 | 2025-08-22 10:15:00 | 578.05 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-08-25 15:15:00 | 582.00 | 2025-08-26 09:15:00 | 575.65 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-09-04 09:15:00 | 583.25 | 2025-09-04 09:15:00 | 570.90 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-10-30 09:30:00 | 569.45 | 2025-11-03 12:15:00 | 575.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-30 11:30:00 | 569.50 | 2025-11-03 12:15:00 | 575.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-31 13:00:00 | 569.30 | 2025-11-03 12:15:00 | 575.70 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-11-03 10:15:00 | 569.25 | 2025-11-03 12:15:00 | 575.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-12-23 14:30:00 | 545.80 | 2026-01-02 13:15:00 | 565.20 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-12-24 15:00:00 | 547.95 | 2026-01-02 13:15:00 | 565.20 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-01-08 13:15:00 | 545.80 | 2026-01-23 13:15:00 | 518.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 546.85 | 2026-01-23 13:15:00 | 519.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 552.80 | 2026-01-23 13:15:00 | 525.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 15:15:00 | 552.65 | 2026-01-23 13:15:00 | 525.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:15:00 | 553.05 | 2026-01-23 13:15:00 | 525.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:45:00 | 553.00 | 2026-01-23 13:15:00 | 525.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 551.30 | 2026-01-23 13:15:00 | 523.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:30:00 | 550.15 | 2026-01-23 13:15:00 | 522.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 10:45:00 | 551.15 | 2026-01-23 13:15:00 | 523.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 545.80 | 2026-02-01 14:15:00 | 497.52 | TARGET_HIT | 0.50 | 8.85% |
| SELL | retest2 | 2026-01-08 15:00:00 | 546.85 | 2026-02-01 14:15:00 | 497.38 | TARGET_HIT | 0.50 | 9.05% |
| SELL | retest2 | 2026-01-16 13:00:00 | 552.80 | 2026-02-01 14:15:00 | 497.74 | TARGET_HIT | 0.50 | 9.96% |
| SELL | retest2 | 2026-01-16 15:15:00 | 552.65 | 2026-02-01 14:15:00 | 497.70 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest2 | 2026-01-19 13:15:00 | 553.05 | 2026-02-01 14:15:00 | 496.17 | TARGET_HIT | 0.50 | 10.28% |
| SELL | retest2 | 2026-01-19 13:45:00 | 553.00 | 2026-02-01 14:15:00 | 496.03 | TARGET_HIT | 0.50 | 10.30% |
| SELL | retest2 | 2026-01-19 15:15:00 | 551.30 | 2026-02-02 09:15:00 | 495.13 | TARGET_HIT | 0.50 | 10.19% |
| SELL | retest2 | 2026-01-20 09:30:00 | 550.15 | 2026-02-09 10:15:00 | 537.80 | STOP_HIT | 0.50 | 2.24% |
| SELL | retest2 | 2026-01-20 10:45:00 | 551.15 | 2026-02-09 10:15:00 | 537.80 | STOP_HIT | 0.50 | 2.42% |
