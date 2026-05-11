# Nava Ltd. (NAVA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 727.65
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 8 |
| TARGET_HIT | 8 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 20
- **Target hits / Stop hits / Partials:** 4 / 26 / 8
- **Avg / median % per leg:** 0.68% / -0.42%
- **Sum % (uncompounded):** 25.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 11.11% | 44.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 11.11% | 44.4% |
| SELL (all) | 34 | 14 | 41.2% | 0 | 26 | 8 | -0.55% | -18.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 14 | 41.2% | 0 | 26 | 8 | -0.55% | -18.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 18 | 47.4% | 4 | 26 | 8 | 0.68% | 25.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 617.45 | 636.93 | 636.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 609.30 | 636.45 | 636.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 559.15 | 550.76 | 577.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 559.15 | 550.76 | 577.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 575.55 | 556.57 | 574.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 573.50 | 556.57 | 574.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 577.20 | 556.78 | 574.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 581.45 | 556.78 | 574.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 576.45 | 557.16 | 574.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 574.50 | 557.76 | 574.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:45:00 | 574.95 | 557.89 | 574.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 574.95 | 558.23 | 574.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:15:00 | 574.45 | 558.40 | 574.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 572.20 | 559.46 | 574.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 572.20 | 559.46 | 574.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 575.00 | 559.62 | 574.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 575.80 | 559.62 | 574.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 566.40 | 559.68 | 574.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 564.60 | 559.74 | 574.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 563.70 | 559.78 | 574.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:00:00 | 564.30 | 559.93 | 574.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 561.55 | 559.24 | 572.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 565.80 | 559.30 | 572.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 558.30 | 559.40 | 572.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 576.90 | 561.67 | 572.24 | SL hit (close>static) qty=1.00 sl=576.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 12:15:00 | 576.05 | 568.01 | 568.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 13:15:00 | 578.00 | 568.11 | 568.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 558.55 | 568.21 | 568.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 10:15:00 | 558.55 | 568.21 | 568.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 558.55 | 568.21 | 568.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 556.75 | 568.21 | 568.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 556.00 | 568.09 | 568.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 553.65 | 568.09 | 568.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 557.55 | 567.98 | 568.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 541.50 | 567.69 | 567.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 15:15:00 | 569.00 | 565.88 | 566.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 15:15:00 | 569.00 | 565.88 | 566.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 569.00 | 565.88 | 566.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 577.50 | 565.88 | 566.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 580.65 | 566.03 | 566.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 585.65 | 566.03 | 566.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 574.25 | 566.11 | 566.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 13:45:00 | 573.15 | 566.39 | 567.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:15:00 | 573.00 | 567.08 | 567.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 544.49 | 566.07 | 566.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 544.35 | 566.07 | 566.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 563.85 | 563.49 | 565.51 | SL hit (close>ema200) qty=0.50 sl=563.49 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 604.20 | 565.01 | 564.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 614.50 | 566.59 | 565.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 15:15:00 | 464.95 | 2025-06-04 09:15:00 | 519.70 | TARGET_HIT | 1.00 | 11.77% |
| BUY | retest2 | 2025-05-13 10:15:00 | 463.60 | 2025-06-04 09:15:00 | 513.98 | TARGET_HIT | 1.00 | 10.87% |
| BUY | retest2 | 2025-05-19 10:15:00 | 463.65 | 2025-06-04 09:15:00 | 517.00 | TARGET_HIT | 1.00 | 11.51% |
| BUY | retest2 | 2025-05-29 09:30:00 | 468.40 | 2025-06-04 09:15:00 | 516.56 | TARGET_HIT | 1.00 | 10.28% |
| SELL | retest2 | 2025-12-22 09:15:00 | 574.50 | 2026-01-05 12:15:00 | 576.90 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-12-22 09:45:00 | 574.95 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-22 11:30:00 | 574.95 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-22 13:15:00 | 574.45 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-12-24 11:15:00 | 564.60 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-12-24 11:45:00 | 563.70 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-12-26 10:00:00 | 564.30 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-12-30 10:30:00 | 561.55 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-12-30 15:00:00 | 558.30 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2026-01-13 13:00:00 | 560.40 | 2026-01-20 15:15:00 | 532.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 560.90 | 2026-01-20 15:15:00 | 532.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 556.60 | 2026-01-21 10:15:00 | 528.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:00:00 | 560.40 | 2026-01-29 15:15:00 | 559.90 | STOP_HIT | 0.50 | 0.09% |
| SELL | retest2 | 2026-01-13 13:45:00 | 560.90 | 2026-01-29 15:15:00 | 559.90 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2026-01-19 09:15:00 | 556.60 | 2026-01-29 15:15:00 | 559.90 | STOP_HIT | 0.50 | -0.59% |
| SELL | retest2 | 2026-02-01 12:15:00 | 553.90 | 2026-02-02 09:15:00 | 526.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 12:15:00 | 553.90 | 2026-02-03 09:15:00 | 565.75 | STOP_HIT | 0.50 | -2.14% |
| SELL | retest2 | 2026-02-13 15:15:00 | 553.00 | 2026-02-24 11:15:00 | 575.50 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2026-02-17 11:00:00 | 554.20 | 2026-02-24 11:15:00 | 575.50 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2026-02-19 10:00:00 | 554.15 | 2026-02-24 11:15:00 | 575.50 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-02-23 10:30:00 | 566.50 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-02-23 12:45:00 | 567.15 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-02-24 09:15:00 | 564.20 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2026-02-24 10:15:00 | 567.05 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2026-03-11 13:45:00 | 573.15 | 2026-03-16 09:15:00 | 544.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 15:15:00 | 573.00 | 2026-03-16 09:15:00 | 544.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:45:00 | 573.15 | 2026-03-18 10:15:00 | 563.85 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2026-03-12 15:15:00 | 573.00 | 2026-03-18 10:15:00 | 563.85 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2026-03-25 10:15:00 | 571.50 | 2026-03-30 09:15:00 | 542.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:00:00 | 574.05 | 2026-03-30 09:15:00 | 545.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 10:15:00 | 571.50 | 2026-04-01 15:15:00 | 562.30 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2026-03-25 11:00:00 | 574.05 | 2026-04-01 15:15:00 | 562.30 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2026-04-02 09:15:00 | 545.05 | 2026-04-08 10:15:00 | 580.75 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2026-04-06 09:45:00 | 551.80 | 2026-04-08 10:15:00 | 580.75 | STOP_HIT | 1.00 | -5.25% |
