# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 561.75
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
| ALERT2_SKIP | 3 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 9 |
| TARGET_HIT | 13 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 30
- **Target hits / Stop hits / Partials:** 13 / 30 / 9
- **Avg / median % per leg:** 1.54% / -0.94%
- **Sum % (uncompounded):** 80.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 5 | 26.3% | 5 | 14 | 0 | 1.25% | 23.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 5 | 26.3% | 5 | 14 | 0 | 1.25% | 23.7% |
| SELL (all) | 33 | 17 | 51.5% | 8 | 16 | 9 | 1.71% | 56.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 17 | 51.5% | 8 | 16 | 9 | 1.71% | 56.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 22 | 42.3% | 13 | 30 | 9 | 1.54% | 80.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 573.80 | 519.88 | 519.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 592.55 | 522.21 | 521.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 531.75 | 535.16 | 528.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 531.75 | 535.16 | 528.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 531.75 | 535.16 | 528.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 532.20 | 535.16 | 528.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 538.80 | 536.11 | 529.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 13:00:00 | 546.45 | 532.78 | 528.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 548.85 | 533.01 | 528.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:30:00 | 547.70 | 533.38 | 528.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 546.00 | 542.89 | 535.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 535.95 | 542.97 | 535.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:15:00 | 538.55 | 542.97 | 535.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-15 12:15:00 | 601.10 | 548.60 | 540.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 519.50 | 548.91 | 548.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 512.30 | 548.54 | 548.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 545.90 | 544.28 | 546.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 545.90 | 544.28 | 546.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 545.90 | 544.28 | 546.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 545.55 | 544.28 | 546.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 542.85 | 544.19 | 546.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 543.50 | 544.19 | 546.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 539.25 | 544.14 | 546.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 536.90 | 544.06 | 546.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 536.40 | 544.06 | 546.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 536.25 | 543.52 | 545.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 537.00 | 543.03 | 545.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 566.65 | 539.29 | 543.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 566.65 | 539.29 | 543.15 | SL hit (close>static) qty=1.00 sl=548.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 587.60 | 547.02 | 546.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 591.75 | 549.01 | 547.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 639.65 | 648.60 | 613.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 15:15:00 | 616.50 | 642.92 | 616.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 616.50 | 642.92 | 616.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 622.30 | 642.92 | 616.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 629.30 | 642.78 | 616.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:30:00 | 632.60 | 641.97 | 616.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 631.95 | 641.97 | 616.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:30:00 | 634.75 | 640.62 | 617.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 610.50 | 640.30 | 621.35 | SL hit (close<static) qty=1.00 sl=615.05 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 563.80 | 614.08 | 614.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 559.40 | 613.04 | 613.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 564.60 | 562.70 | 580.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:45:00 | 566.50 | 562.70 | 580.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 575.55 | 558.37 | 574.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 575.05 | 558.37 | 574.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 583.80 | 558.62 | 574.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 583.80 | 558.62 | 574.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 572.80 | 565.94 | 576.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 577.60 | 565.94 | 576.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 566.60 | 564.59 | 575.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 561.75 | 564.56 | 574.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 560.20 | 564.56 | 574.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 560.45 | 564.51 | 574.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 559.80 | 564.44 | 574.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 533.66 | 561.61 | 572.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 532.19 | 561.61 | 572.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 532.43 | 561.61 | 572.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 531.81 | 561.61 | 572.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 505.57 | 553.44 | 566.60 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-20 14:15:00 | 504.25 | 2025-05-21 10:15:00 | 518.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-05-21 09:15:00 | 501.45 | 2025-05-21 10:15:00 | 518.00 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-05-21 09:45:00 | 505.15 | 2025-05-21 10:15:00 | 518.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-24 13:00:00 | 546.45 | 2025-07-15 12:15:00 | 601.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 09:15:00 | 548.85 | 2025-07-15 12:15:00 | 603.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 11:30:00 | 547.70 | 2025-07-15 12:15:00 | 602.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-03 15:15:00 | 546.00 | 2025-07-15 12:15:00 | 600.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-08 13:15:00 | 538.55 | 2025-07-15 12:15:00 | 592.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:00:00 | 540.50 | 2025-08-08 15:15:00 | 533.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-11 09:15:00 | 538.20 | 2025-08-11 09:15:00 | 529.10 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-08-11 10:45:00 | 538.75 | 2025-08-14 09:15:00 | 533.25 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-08-21 14:30:00 | 554.15 | 2025-08-22 14:15:00 | 546.85 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-22 11:00:00 | 550.65 | 2025-08-22 14:15:00 | 546.85 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-22 11:45:00 | 551.15 | 2025-08-22 14:15:00 | 546.85 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-09-04 10:30:00 | 536.90 | 2025-09-15 09:15:00 | 566.65 | STOP_HIT | 1.00 | -5.54% |
| SELL | retest2 | 2025-09-04 11:00:00 | 536.40 | 2025-09-15 09:15:00 | 566.65 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-09-05 10:45:00 | 536.25 | 2025-09-15 09:15:00 | 566.65 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2025-09-08 09:15:00 | 537.00 | 2025-09-15 09:15:00 | 566.65 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2025-10-27 14:30:00 | 632.60 | 2025-11-07 09:15:00 | 610.50 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-10-27 15:15:00 | 631.95 | 2025-11-07 09:15:00 | 610.50 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-10-29 10:30:00 | 634.75 | 2025-11-07 09:15:00 | 610.50 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2025-11-10 09:15:00 | 633.30 | 2025-11-11 10:15:00 | 613.55 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-11-14 11:15:00 | 625.20 | 2025-11-14 12:15:00 | 616.35 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-11-17 09:15:00 | 632.25 | 2025-11-19 09:15:00 | 617.95 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-11-20 09:30:00 | 625.10 | 2025-11-20 12:15:00 | 619.25 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-11-20 10:30:00 | 626.00 | 2025-11-20 12:15:00 | 619.25 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-13 11:30:00 | 561.75 | 2026-01-20 09:15:00 | 533.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 560.20 | 2026-01-20 09:15:00 | 532.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 560.45 | 2026-01-20 09:15:00 | 532.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 559.80 | 2026-01-20 09:15:00 | 531.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:30:00 | 561.75 | 2026-01-23 13:15:00 | 505.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 560.20 | 2026-01-23 13:15:00 | 504.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 560.45 | 2026-01-23 13:15:00 | 504.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 559.80 | 2026-01-23 13:15:00 | 503.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 10:00:00 | 535.75 | 2026-02-19 09:15:00 | 570.75 | STOP_HIT | 1.00 | -6.53% |
| SELL | retest2 | 2026-02-13 14:00:00 | 535.40 | 2026-02-19 09:15:00 | 570.75 | STOP_HIT | 1.00 | -6.60% |
| SELL | retest2 | 2026-02-16 10:15:00 | 536.40 | 2026-02-19 09:15:00 | 570.75 | STOP_HIT | 1.00 | -6.40% |
| SELL | retest2 | 2026-02-16 11:15:00 | 536.65 | 2026-02-19 09:15:00 | 570.75 | STOP_HIT | 1.00 | -6.35% |
| SELL | retest2 | 2026-02-19 15:15:00 | 544.20 | 2026-03-02 09:15:00 | 516.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:15:00 | 545.80 | 2026-03-02 09:15:00 | 518.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 545.20 | 2026-03-02 09:15:00 | 517.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-02 09:15:00 | 515.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 15:15:00 | 544.20 | 2026-03-04 09:15:00 | 491.22 | TARGET_HIT | 0.50 | 9.74% |
| SELL | retest2 | 2026-02-23 10:15:00 | 545.80 | 2026-03-04 11:15:00 | 489.78 | TARGET_HIT | 0.50 | 10.26% |
| SELL | retest2 | 2026-02-23 11:15:00 | 545.20 | 2026-03-04 11:15:00 | 490.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-04 13:15:00 | 488.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 14:30:00 | 489.05 | 2026-04-24 15:15:00 | 464.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 14:30:00 | 489.05 | 2026-04-27 10:15:00 | 493.25 | STOP_HIT | 0.50 | -0.86% |
| SELL | retest2 | 2026-04-27 11:15:00 | 490.65 | 2026-04-27 13:15:00 | 504.30 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-04-28 15:00:00 | 492.90 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-04-29 13:30:00 | 492.80 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-05-04 12:00:00 | 505.15 | 2026-05-05 09:15:00 | 515.30 | STOP_HIT | 1.00 | -2.01% |
