# ICICI Prudential Life Insurance Company Ltd. (ICICIPRULI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 565.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 18 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 16
- **Target hits / Stop hits / Partials:** 0 / 26 / 11
- **Avg / median % per leg:** 1.02% / 1.32%
- **Sum % (uncompounded):** 37.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 0 | 14 | 4 | 0.08% | 1.4% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.19% | 25.5% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.41% | -24.1% |
| SELL (all) | 19 | 13 | 68.4% | 0 | 12 | 7 | 1.91% | 36.3% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.73% | 29.9% |
| SELL @ 3rd Alert (retest2) | 11 | 5 | 45.5% | 0 | 8 | 3 | 0.58% | 6.4% |
| retest1 (combined) | 16 | 16 | 100.0% | 0 | 8 | 8 | 3.46% | 55.4% |
| retest2 (combined) | 21 | 5 | 23.8% | 0 | 18 | 3 | -0.84% | -17.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 613.30 | 591.81 | 591.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 617.20 | 593.07 | 592.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 627.00 | 629.27 | 616.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 14:30:00 | 628.20 | 629.11 | 616.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 15:00:00 | 629.65 | 629.11 | 616.85 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 13:15:00 | 631.20 | 630.29 | 618.58 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 12:00:00 | 628.20 | 630.29 | 618.92 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 659.61 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 661.13 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 662.76 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:15:00 | 659.61 | 634.70 | 623.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 637.95 | 638.26 | 626.86 | SL hit (close<ema200) qty=0.50 sl=638.26 alert=retest1 |

### Cycle 2 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 615.00 | 630.43 | 630.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 13:15:00 | 612.35 | 629.36 | 629.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 10:15:00 | 628.90 | 625.95 | 628.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 10:15:00 | 628.90 | 625.95 | 628.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 628.90 | 625.95 | 628.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 628.90 | 625.95 | 628.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 624.45 | 625.93 | 628.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:30:00 | 623.05 | 625.87 | 627.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 621.50 | 625.83 | 627.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 629.90 | 625.82 | 627.91 | SL hit (close>static) qty=1.00 sl=629.30 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 630.35 | 608.33 | 608.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 631.00 | 614.74 | 612.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 15:15:00 | 660.50 | 660.98 | 645.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:15:00 | 662.70 | 660.98 | 645.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 645.85 | 660.36 | 645.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 647.85 | 660.36 | 645.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 645.00 | 660.21 | 645.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:45:00 | 641.85 | 660.21 | 645.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 639.65 | 659.85 | 645.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:45:00 | 639.85 | 659.85 | 645.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 649.70 | 659.03 | 645.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 655.30 | 650.87 | 644.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:45:00 | 654.40 | 652.12 | 645.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 653.10 | 652.10 | 645.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:30:00 | 654.65 | 651.98 | 645.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 642.50 | 651.76 | 645.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 642.50 | 651.76 | 645.70 | SL hit (close<static) qty=1.00 sl=643.15 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 600.60 | 644.37 | 644.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 596.40 | 638.86 | 641.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 565.10 | 564.36 | 591.77 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 12:00:00 | 562.45 | 564.36 | 591.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 13:00:00 | 561.05 | 564.32 | 591.34 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:45:00 | 561.75 | 564.23 | 590.63 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 12:15:00 | 563.00 | 564.23 | 590.50 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 534.33 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 533.00 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 533.66 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:15:00 | 534.85 | 560.37 | 584.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 548.20 | 545.59 | 569.85 | SL hit (close>ema200) qty=0.50 sl=545.59 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-13 14:30:00 | 628.20 | 2025-07-01 09:15:00 | 659.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-13 15:00:00 | 629.65 | 2025-07-01 09:15:00 | 661.13 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-18 13:15:00 | 631.20 | 2025-07-01 09:15:00 | 662.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-19 12:00:00 | 628.20 | 2025-07-01 09:15:00 | 659.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-13 14:30:00 | 628.20 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2025-06-13 15:00:00 | 629.65 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-06-18 13:15:00 | 631.20 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2025-06-19 12:00:00 | 628.20 | 2025-07-03 15:15:00 | 637.95 | STOP_HIT | 0.50 | 1.55% |
| BUY | retest2 | 2025-07-24 09:15:00 | 631.35 | 2025-07-24 14:15:00 | 624.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-08-13 13:30:00 | 623.05 | 2025-08-14 09:15:00 | 629.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-13 15:00:00 | 621.50 | 2025-08-14 09:15:00 | 629.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-22 11:15:00 | 623.15 | 2025-08-25 12:15:00 | 629.55 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-08-26 09:15:00 | 622.30 | 2025-09-08 09:15:00 | 591.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 622.30 | 2025-09-22 09:15:00 | 612.80 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-09-04 11:30:00 | 609.05 | 2025-10-15 10:15:00 | 580.45 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-09-22 11:15:00 | 611.00 | 2025-10-15 10:15:00 | 580.16 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-09-04 11:30:00 | 609.05 | 2025-10-23 09:15:00 | 609.75 | STOP_HIT | 0.50 | -0.11% |
| SELL | retest2 | 2025-09-22 11:15:00 | 611.00 | 2025-10-23 09:15:00 | 609.75 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2025-09-22 14:30:00 | 610.70 | 2025-11-14 15:15:00 | 630.35 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-11-11 09:15:00 | 610.60 | 2025-11-14 15:15:00 | 630.35 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-02-03 09:15:00 | 655.30 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-06 09:45:00 | 654.40 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-02-06 11:15:00 | 653.10 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-02-09 10:30:00 | 654.65 | 2026-02-10 09:15:00 | 642.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-02-19 10:30:00 | 655.10 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2026-02-19 13:00:00 | 651.55 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-02-19 14:45:00 | 651.25 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-02-20 09:30:00 | 653.25 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-03-02 11:00:00 | 656.95 | 2026-03-04 09:15:00 | 633.10 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest1 | 2026-04-15 12:00:00 | 562.45 | 2026-04-23 09:15:00 | 534.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-15 13:00:00 | 561.05 | 2026-04-23 09:15:00 | 533.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-16 10:45:00 | 561.75 | 2026-04-23 09:15:00 | 533.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-16 12:15:00 | 563.00 | 2026-04-23 09:15:00 | 534.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-15 12:00:00 | 562.45 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2026-04-15 13:00:00 | 561.05 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest1 | 2026-04-16 10:45:00 | 561.75 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest1 | 2026-04-16 12:15:00 | 563.00 | 2026-05-06 11:15:00 | 548.20 | STOP_HIT | 0.50 | 2.63% |
