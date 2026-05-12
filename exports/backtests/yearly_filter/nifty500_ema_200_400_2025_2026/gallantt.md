# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 866.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 13 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 4
- **Target hits / Stop hits / Partials:** 9 / 5 / 6
- **Avg / median % per leg:** 5.81% / 5.59%
- **Sum % (uncompounded):** 116.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 5 | 4 | 1 | 4.57% | 45.7% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 4 | 4 | 0 | 3.84% | 30.7% |
| SELL (all) | 10 | 10 | 100.0% | 4 | 1 | 5 | 7.06% | 70.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 10 | 100.0% | 4 | 1 | 5 | 7.06% | 70.6% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 18 | 14 | 77.8% | 8 | 5 | 5 | 5.63% | 101.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 09:15:00 | 435.05 | 423.64 | 395.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 456.80 | 424.23 | 396.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-05-13 09:15:00 | 478.56 | 426.54 | 398.45 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 517.00 | 633.70 | 633.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 506.20 | 591.07 | 600.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 563.20 | 558.16 | 577.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 563.20 | 558.16 | 577.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 573.35 | 560.04 | 575.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 567.10 | 561.02 | 575.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:30:00 | 566.60 | 561.13 | 575.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 562.65 | 560.32 | 573.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:45:00 | 566.05 | 560.49 | 573.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:15:00 | 538.75 | 558.84 | 571.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:15:00 | 538.27 | 558.84 | 571.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:15:00 | 537.75 | 558.84 | 571.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:15:00 | 534.52 | 557.78 | 570.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-30 09:15:00 | 510.39 | 549.98 | 564.72 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 698.50 | 559.18 | 558.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 722.05 | 588.32 | 574.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 09:15:00 | 435.05 | 2025-05-12 10:15:00 | 456.80 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-12 09:15:00 | 435.05 | 2025-05-13 09:15:00 | 478.56 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-03 11:45:00 | 619.10 | 2025-09-08 10:15:00 | 681.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-03 15:15:00 | 619.75 | 2025-09-08 10:15:00 | 681.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-04 10:15:00 | 621.20 | 2025-09-08 10:15:00 | 683.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 09:15:00 | 625.15 | 2025-09-08 10:15:00 | 687.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 09:15:00 | 653.40 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-10-15 13:00:00 | 651.20 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-15 14:45:00 | 656.70 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-10-17 14:00:00 | 651.90 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-01-12 09:15:00 | 567.10 | 2026-01-22 11:15:00 | 538.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 10:30:00 | 566.60 | 2026-01-22 11:15:00 | 538.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 562.65 | 2026-01-22 11:15:00 | 537.75 | PARTIAL | 0.50 | 4.43% |
| SELL | retest2 | 2026-01-20 09:45:00 | 566.05 | 2026-01-23 10:15:00 | 534.52 | PARTIAL | 0.50 | 5.57% |
| SELL | retest2 | 2026-01-12 09:15:00 | 567.10 | 2026-01-30 09:15:00 | 510.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-12 10:30:00 | 566.60 | 2026-01-30 09:15:00 | 509.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 562.65 | 2026-01-30 09:15:00 | 509.44 | TARGET_HIT | 0.50 | 9.46% |
| SELL | retest2 | 2026-01-20 09:45:00 | 566.05 | 2026-02-02 10:15:00 | 506.38 | TARGET_HIT | 0.50 | 10.54% |
| SELL | retest2 | 2026-02-06 15:00:00 | 587.00 | 2026-02-09 09:15:00 | 557.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 15:00:00 | 587.00 | 2026-02-09 09:15:00 | 554.20 | STOP_HIT | 0.50 | 5.59% |
