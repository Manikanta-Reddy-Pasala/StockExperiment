# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 596.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 0 |
| TARGET_HIT | 10 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 18
- **Target hits / Stop hits / Partials:** 10 / 18 / 0
- **Avg / median % per leg:** 2.20% / -0.80%
- **Sum % (uncompounded):** 61.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 10 | 37.0% | 10 | 17 | 0 | 2.34% | 63.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 10 | 37.0% | 10 | 17 | 0 | 2.34% | 63.1% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.54% | -1.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.54% | -1.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 10 | 35.7% | 10 | 18 | 0 | 2.20% | 61.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 461.20 | 505.40 | 505.44 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 519.20 | 504.12 | 504.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 526.00 | 505.15 | 504.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 506.70 | 510.39 | 507.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 506.70 | 510.39 | 507.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 506.70 | 510.39 | 507.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 506.70 | 510.39 | 507.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 506.20 | 510.35 | 507.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 506.20 | 510.35 | 507.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 507.00 | 510.32 | 507.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 514.25 | 510.32 | 507.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 12:15:00 | 509.50 | 510.34 | 507.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 508.00 | 510.30 | 507.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 509.15 | 510.28 | 507.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 506.25 | 510.24 | 507.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:45:00 | 506.05 | 510.24 | 507.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 505.10 | 510.19 | 507.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 505.10 | 510.19 | 507.69 | SL hit (close<static) qty=1.00 sl=505.15 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 524.80 | 549.78 | 549.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 523.65 | 549.52 | 549.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 547.70 | 547.16 | 548.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 547.70 | 547.16 | 548.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 547.70 | 547.16 | 548.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 552.40 | 547.16 | 548.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 551.20 | 547.20 | 548.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 551.20 | 547.20 | 548.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 558.20 | 547.31 | 548.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:00:00 | 558.20 | 547.31 | 548.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 548.85 | 547.62 | 548.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:00:00 | 545.50 | 547.68 | 548.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 553.90 | 547.70 | 548.70 | SL hit (close>static) qty=1.00 sl=552.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 569.00 | 549.66 | 549.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 576.25 | 549.92 | 549.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 553.80 | 556.55 | 553.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 553.80 | 556.55 | 553.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 553.80 | 556.55 | 553.33 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 10:15:00 | 516.60 | 550.41 | 550.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 504.40 | 545.45 | 547.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 09:15:00 | 564.00 | 522.48 | 533.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 564.00 | 522.48 | 533.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 564.00 | 522.48 | 533.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 560.60 | 522.48 | 533.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 570.70 | 522.96 | 534.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 571.95 | 522.96 | 534.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 590.70 | 541.66 | 541.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 607.40 | 544.19 | 542.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 588.55 | 599.54 | 575.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:30:00 | 596.45 | 599.54 | 575.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 451.50 | 2025-05-19 09:15:00 | 496.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 09:15:00 | 514.25 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-09-29 12:15:00 | 509.50 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-30 09:15:00 | 508.00 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-30 09:45:00 | 509.15 | 2025-09-30 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-13 14:00:00 | 517.30 | 2025-10-29 10:15:00 | 569.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 09:30:00 | 516.75 | 2025-10-29 10:15:00 | 568.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 11:15:00 | 515.55 | 2025-10-29 10:15:00 | 567.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 12:00:00 | 520.00 | 2025-10-29 10:15:00 | 561.22 | TARGET_HIT | 1.00 | 7.93% |
| BUY | retest2 | 2025-10-20 13:15:00 | 510.20 | 2025-10-29 10:15:00 | 563.59 | TARGET_HIT | 1.00 | 10.46% |
| BUY | retest2 | 2025-10-21 13:45:00 | 512.35 | 2025-10-29 12:15:00 | 572.00 | TARGET_HIT | 1.00 | 11.64% |
| BUY | retest2 | 2025-11-17 13:30:00 | 510.00 | 2025-11-18 09:15:00 | 506.10 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-11-17 14:00:00 | 509.95 | 2025-11-18 09:15:00 | 506.10 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-26 11:00:00 | 533.25 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-11-26 12:00:00 | 538.15 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-11-28 09:30:00 | 532.15 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-11-28 11:15:00 | 532.70 | 2025-12-05 10:15:00 | 518.60 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-12-03 09:30:00 | 525.25 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-03 11:30:00 | 524.95 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-12-04 09:45:00 | 526.65 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-12-04 13:15:00 | 525.25 | 2025-12-08 11:15:00 | 515.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-22 09:15:00 | 533.90 | 2025-12-29 09:15:00 | 587.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-22 12:15:00 | 532.20 | 2025-12-29 09:15:00 | 585.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 09:45:00 | 532.70 | 2025-12-29 09:15:00 | 585.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-23 14:00:00 | 532.80 | 2026-02-01 13:15:00 | 522.95 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-02-10 12:15:00 | 564.85 | 2026-02-11 11:15:00 | 534.50 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2026-02-11 09:15:00 | 564.40 | 2026-02-11 11:15:00 | 534.50 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2026-02-19 15:00:00 | 545.50 | 2026-02-20 09:15:00 | 553.90 | STOP_HIT | 1.00 | -1.54% |
