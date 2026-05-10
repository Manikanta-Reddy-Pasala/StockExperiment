# PG Electroplast Ltd. (PGEL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 530.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 4
- **Avg / median % per leg:** 0.40% / -1.99%
- **Sum % (uncompounded):** 7.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.81% | -5.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.81% | -5.6% |
| SELL (all) | 18 | 8 | 44.4% | 4 | 10 | 4 | 0.76% | 13.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 4 | 10 | 4 | 0.76% | 13.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 8 | 40.0% | 4 | 12 | 4 | 0.40% | 8.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 793.90 | 859.41 | 859.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 772.00 | 848.96 | 854.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 768.25 | 763.95 | 787.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:00:00 | 768.25 | 763.95 | 787.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 789.90 | 764.66 | 786.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 789.90 | 764.66 | 786.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 789.00 | 764.91 | 786.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 784.50 | 764.91 | 786.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 782.25 | 765.05 | 786.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 783.00 | 766.22 | 785.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 803.65 | 766.95 | 785.68 | SL hit (close>static) qty=1.00 sl=796.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 803.65 | 766.95 | 785.68 | SL hit (close>static) qty=1.00 sl=796.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 803.65 | 766.95 | 785.68 | SL hit (close>static) qty=1.00 sl=796.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 782.90 | 782.26 | 790.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 787.15 | 782.31 | 790.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:15:00 | 794.35 | 782.31 | 790.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 791.95 | 782.41 | 790.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 14:15:00 | 783.35 | 782.57 | 790.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 798.90 | 782.96 | 790.76 | SL hit (close>static) qty=1.00 sl=796.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 798.90 | 782.96 | 790.76 | SL hit (close>static) qty=1.00 sl=794.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 780.85 | 790.26 | 793.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:00:00 | 780.45 | 790.04 | 793.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 781.70 | 789.96 | 793.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 792.50 | 789.88 | 792.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 792.65 | 789.88 | 792.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 790.00 | 789.88 | 792.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 768.45 | 789.87 | 792.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 741.81 | 787.20 | 791.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 741.43 | 787.20 | 791.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 742.62 | 787.20 | 791.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 730.03 | 786.62 | 791.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-08 13:15:00 | 702.76 | 780.31 | 787.57 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-08 13:15:00 | 702.41 | 780.31 | 787.57 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-08 13:15:00 | 703.53 | 780.31 | 787.57 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-08 13:15:00 | 691.61 | 780.31 | 787.57 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 628.30 | 578.84 | 578.63 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 10:15:00 | 533.65 | 580.06 | 580.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 11:15:00 | 528.85 | 579.55 | 580.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 562.60 | 561.82 | 569.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 562.60 | 561.82 | 569.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 582.45 | 562.04 | 569.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:00:00 | 571.35 | 562.14 | 569.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:45:00 | 578.20 | 562.13 | 569.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 578.60 | 562.13 | 569.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:45:00 | 579.70 | 562.29 | 569.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 569.20 | 565.09 | 570.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 624.35 | 570.77 | 572.95 | SL hit (close>static) qty=1.00 sl=609.20 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 624.25 | 575.15 | 575.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 628.65 | 584.32 | 579.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 597.30 | 599.95 | 590.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 597.30 | 599.95 | 590.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 597.30 | 599.95 | 590.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:30:00 | 597.10 | 599.95 | 590.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 582.35 | 599.77 | 590.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:00:00 | 582.35 | 599.77 | 590.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 577.90 | 599.56 | 590.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:45:00 | 578.10 | 599.56 | 590.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 592.65 | 599.15 | 590.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 601.60 | 599.15 | 590.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 589.55 | 598.95 | 590.09 | SL hit (close<static) qty=1.00 sl=589.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 601.45 | 598.99 | 590.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 579.70 | 599.93 | 591.11 | SL hit (close<static) qty=1.00 sl=589.60 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 513.05 | 583.49 | 583.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 505.40 | 582.71 | 583.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 10:15:00 | 517.35 | 511.44 | 537.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 10:45:00 | 515.00 | 511.44 | 537.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 540.90 | 512.09 | 537.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 540.90 | 512.09 | 537.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 537.55 | 512.34 | 537.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 541.70 | 512.34 | 537.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 537.05 | 512.59 | 537.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:15:00 | 541.95 | 512.59 | 537.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 545.00 | 512.91 | 537.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 544.50 | 512.91 | 537.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 538.60 | 535.20 | 542.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 541.05 | 535.20 | 542.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 539.00 | 535.10 | 542.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:00:00 | 535.00 | 535.12 | 542.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 544.00 | 535.25 | 542.24 | SL hit (close>static) qty=1.00 sl=542.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 535.05 | 535.35 | 542.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:00:00 | 534.00 | 535.34 | 542.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 534.85 | 535.40 | 542.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-10 09:15:00 | 784.50 | 2025-07-15 09:15:00 | 803.65 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-07-10 09:45:00 | 782.25 | 2025-07-15 09:15:00 | 803.65 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-07-14 13:45:00 | 783.00 | 2025-07-15 09:15:00 | 803.65 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-23 09:15:00 | 782.90 | 2025-07-24 09:15:00 | 798.90 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-23 14:15:00 | 783.35 | 2025-07-24 09:15:00 | 798.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-08-01 14:30:00 | 780.85 | 2025-08-07 09:15:00 | 741.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 10:00:00 | 780.45 | 2025-08-07 09:15:00 | 741.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 10:45:00 | 781.70 | 2025-08-07 09:15:00 | 742.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 768.45 | 2025-08-07 10:15:00 | 730.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-01 14:30:00 | 780.85 | 2025-08-08 13:15:00 | 702.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-04 10:00:00 | 780.45 | 2025-08-08 13:15:00 | 702.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-04 10:45:00 | 781.70 | 2025-08-08 13:15:00 | 703.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 768.45 | 2025-08-08 13:15:00 | 691.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 11:00:00 | 571.35 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -9.28% |
| SELL | retest2 | 2026-02-04 09:45:00 | 578.20 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -7.98% |
| SELL | retest2 | 2026-02-04 10:15:00 | 578.60 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -7.91% |
| SELL | retest2 | 2026-02-04 10:45:00 | 579.70 | 2026-02-11 10:15:00 | 624.35 | STOP_HIT | 1.00 | -7.70% |
| BUY | retest2 | 2026-03-05 09:15:00 | 601.60 | 2026-03-05 11:15:00 | 589.55 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-03-05 12:45:00 | 601.45 | 2026-03-09 09:15:00 | 579.70 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-05-06 12:00:00 | 535.00 | 2026-05-06 14:15:00 | 544.00 | STOP_HIT | 1.00 | -1.68% |
