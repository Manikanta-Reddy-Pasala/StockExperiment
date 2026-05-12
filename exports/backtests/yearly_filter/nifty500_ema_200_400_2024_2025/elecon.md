# Elecon Engineering Co. Ltd. (ELECON)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 562.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 5 |
| TARGET_HIT | 6 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 9
- **Target hits / Stop hits / Partials:** 6 / 10 / 5
- **Avg / median % per leg:** 2.85% / 5.00%
- **Sum % (uncompounded):** 59.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 2 | 2 | 0 | 3.60% | 14.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 2 | 2 | 0 | 3.60% | 14.4% |
| SELL (all) | 17 | 10 | 58.8% | 4 | 8 | 5 | 2.67% | 45.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 10 | 58.8% | 4 | 8 | 5 | 2.67% | 45.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 12 | 57.1% | 6 | 10 | 5 | 2.85% | 59.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 547.25 | 633.69 | 633.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 13:15:00 | 542.50 | 632.79 | 633.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 595.95 | 587.52 | 604.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 595.95 | 587.52 | 604.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 621.50 | 586.31 | 600.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 621.50 | 586.31 | 600.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 616.40 | 586.61 | 600.96 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 649.35 | 611.34 | 611.25 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 551.35 | 614.05 | 614.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 15:15:00 | 549.00 | 613.41 | 613.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 14:15:00 | 617.90 | 607.97 | 611.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 14:15:00 | 617.90 | 607.97 | 611.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 617.90 | 607.97 | 611.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 617.90 | 607.97 | 611.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 613.50 | 608.02 | 611.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 609.90 | 608.02 | 611.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 14:30:00 | 609.20 | 606.13 | 609.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:30:00 | 605.05 | 606.10 | 609.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:15:00 | 579.40 | 605.42 | 609.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:15:00 | 578.74 | 605.42 | 609.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:15:00 | 574.80 | 605.42 | 609.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-22 13:15:00 | 548.91 | 604.81 | 608.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 567.35 | 484.39 | 484.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 13:15:00 | 575.60 | 485.30 | 484.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 653.05 | 654.45 | 607.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:00:00 | 653.05 | 654.45 | 607.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 617.30 | 646.62 | 625.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 619.25 | 646.62 | 625.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 617.20 | 646.32 | 625.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:15:00 | 618.00 | 646.32 | 625.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 618.00 | 646.04 | 625.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 627.40 | 646.04 | 625.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 613.90 | 643.23 | 625.56 | SL hit (close<static) qty=1.00 sl=615.40 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 578.90 | 614.33 | 614.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 570.90 | 613.55 | 613.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 581.05 | 580.80 | 593.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:00:00 | 581.05 | 580.80 | 593.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 592.30 | 567.91 | 580.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 594.40 | 567.91 | 580.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 605.45 | 568.29 | 580.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 605.45 | 568.29 | 580.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 605.00 | 589.65 | 589.60 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 575.35 | 589.46 | 589.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 569.50 | 589.26 | 589.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 587.70 | 585.08 | 587.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 587.70 | 585.08 | 587.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 587.70 | 585.08 | 587.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 582.60 | 585.08 | 587.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 584.20 | 585.07 | 587.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 583.85 | 585.07 | 587.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 585.55 | 585.04 | 587.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 585.55 | 585.04 | 587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 586.70 | 585.05 | 587.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:00:00 | 585.35 | 585.08 | 587.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 583.60 | 585.05 | 587.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 15:00:00 | 585.10 | 585.01 | 587.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 12:15:00 | 589.35 | 585.12 | 587.06 | SL hit (close>static) qty=1.00 sl=589.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 607.40 | 589.00 | 588.92 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 552.20 | 588.66 | 588.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 537.05 | 585.64 | 587.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 569.50 | 569.13 | 576.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 15:00:00 | 569.50 | 569.13 | 576.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 508.60 | 491.24 | 511.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 508.60 | 491.24 | 511.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 500.65 | 491.34 | 511.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 499.40 | 491.43 | 511.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 499.45 | 491.63 | 511.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 499.15 | 491.93 | 511.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 519.25 | 492.10 | 510.19 | SL hit (close>static) qty=1.00 sl=516.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 509.45 | 424.66 | 424.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 510.05 | 426.33 | 425.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 14:30:00 | 501.45 | 2024-05-14 15:15:00 | 551.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 508.83 | 2024-06-05 09:15:00 | 491.28 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-06-05 10:45:00 | 505.40 | 2024-06-07 13:15:00 | 555.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-16 09:15:00 | 609.90 | 2025-01-22 12:15:00 | 579.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 14:30:00 | 609.20 | 2025-01-22 12:15:00 | 578.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 09:30:00 | 605.05 | 2025-01-22 12:15:00 | 574.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 09:15:00 | 609.90 | 2025-01-22 13:15:00 | 548.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 14:30:00 | 609.20 | 2025-01-22 13:15:00 | 548.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 09:30:00 | 605.05 | 2025-01-22 13:15:00 | 544.54 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-14 09:15:00 | 627.40 | 2025-07-16 09:15:00 | 613.90 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-10-03 10:00:00 | 585.35 | 2025-10-06 12:15:00 | 589.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-03 12:00:00 | 583.60 | 2025-10-06 12:15:00 | 589.35 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-03 15:00:00 | 585.10 | 2025-10-06 12:15:00 | 589.35 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-02 13:15:00 | 499.40 | 2026-01-07 11:15:00 | 519.25 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-01-02 14:45:00 | 499.45 | 2026-01-07 11:15:00 | 519.25 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-01-05 11:15:00 | 499.15 | 2026-01-07 11:15:00 | 519.25 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-01-09 09:15:00 | 437.50 | 2026-01-09 11:15:00 | 415.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 437.50 | 2026-01-14 11:15:00 | 393.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-05 09:15:00 | 446.00 | 2026-02-09 10:15:00 | 484.25 | STOP_HIT | 1.00 | -8.58% |
| SELL | retest2 | 2026-02-12 09:15:00 | 452.30 | 2026-02-16 09:15:00 | 429.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 452.30 | 2026-02-16 15:15:00 | 436.90 | STOP_HIT | 0.50 | 3.40% |
