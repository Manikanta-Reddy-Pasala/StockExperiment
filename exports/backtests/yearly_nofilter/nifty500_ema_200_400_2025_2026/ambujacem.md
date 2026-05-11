# Ambuja Cements Ltd. (AMBUJACEM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 443.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 9 |
| TARGET_HIT | 10 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 16
- **Target hits / Stop hits / Partials:** 10 / 18 / 9
- **Avg / median % per leg:** 3.22% / 5.00%
- **Sum % (uncompounded):** 119.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 3 | 23.1% | 3 | 10 | 0 | 0.93% | 12.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 3 | 23.1% | 3 | 10 | 0 | 0.93% | 12.1% |
| SELL (all) | 24 | 18 | 75.0% | 7 | 8 | 9 | 4.46% | 107.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 18 | 75.0% | 7 | 8 | 9 | 4.46% | 107.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 21 | 56.8% | 10 | 18 | 9 | 3.22% | 119.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 552.25 | 577.87 | 577.92 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 590.50 | 577.72 | 577.69 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-09-30 09:15:00)

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
