# HBL Engineering Ltd. (HBLENGINE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 850.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 42 |
| ALERT2 | 40 |
| ALERT2_SKIP | 18 |
| ALERT3 | 104 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 61 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 43
- **Target hits / Stop hits / Partials:** 0 / 62 / 6
- **Avg / median % per leg:** 0.23% / -0.76%
- **Sum % (uncompounded):** 15.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 10 | 30.3% | 0 | 33 | 0 | 0.04% | 1.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.00% | -2.0% |
| BUY @ 3rd Alert (retest2) | 32 | 10 | 31.2% | 0 | 32 | 0 | 0.11% | 3.4% |
| SELL (all) | 35 | 15 | 42.9% | 0 | 29 | 6 | 0.41% | 14.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 15 | 42.9% | 0 | 29 | 6 | 0.41% | 14.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.00% | -2.0% |
| retest2 (combined) | 67 | 25 | 37.3% | 0 | 61 | 6 | 0.27% | 17.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 496.20 | 477.50 | 476.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 506.00 | 486.06 | 480.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 569.70 | 570.80 | 559.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 14:30:00 | 567.55 | 570.80 | 559.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 574.25 | 581.87 | 573.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 574.25 | 581.87 | 573.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 581.70 | 581.84 | 574.16 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 567.75 | 572.29 | 572.84 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 575.65 | 573.16 | 573.15 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 571.00 | 572.73 | 572.95 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 577.30 | 572.75 | 572.31 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 559.70 | 570.35 | 571.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 557.75 | 567.83 | 570.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 562.30 | 560.80 | 564.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 11:00:00 | 562.30 | 560.80 | 564.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 567.25 | 562.09 | 565.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:00:00 | 561.70 | 562.62 | 564.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 572.20 | 564.81 | 565.32 | SL hit (close>static) qty=1.00 sl=571.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 573.50 | 566.55 | 566.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 581.30 | 571.39 | 568.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 13:15:00 | 601.50 | 604.30 | 595.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 14:00:00 | 601.50 | 604.30 | 595.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 605.00 | 604.10 | 597.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:00:00 | 609.95 | 600.44 | 598.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:45:00 | 608.45 | 604.22 | 600.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 608.90 | 604.22 | 600.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 609.00 | 609.79 | 606.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 605.90 | 609.01 | 606.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 606.35 | 609.01 | 606.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 606.75 | 608.56 | 606.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 608.80 | 608.56 | 606.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 608.80 | 607.42 | 606.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 608.75 | 608.01 | 607.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 608.80 | 607.96 | 607.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 603.50 | 607.13 | 606.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-09 14:15:00 | 603.50 | 607.13 | 606.96 | SL hit (close<static) qty=1.00 sl=604.50 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 604.20 | 606.54 | 606.71 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 610.65 | 606.99 | 606.86 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 605.00 | 606.59 | 606.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 14:15:00 | 604.00 | 605.71 | 606.26 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 618.05 | 608.05 | 607.22 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 598.85 | 606.82 | 607.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 592.65 | 602.28 | 605.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 602.50 | 595.07 | 598.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 602.50 | 595.07 | 598.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 602.50 | 595.07 | 598.92 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 605.25 | 600.56 | 600.48 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 597.15 | 600.30 | 600.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 596.15 | 599.47 | 600.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 577.35 | 571.38 | 579.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:00:00 | 577.35 | 571.38 | 579.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 580.25 | 573.15 | 579.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 579.40 | 573.15 | 579.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 577.55 | 574.03 | 579.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 575.20 | 574.03 | 579.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:00:00 | 576.20 | 574.47 | 578.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:30:00 | 576.70 | 575.07 | 578.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 574.00 | 576.56 | 578.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 588.30 | 574.26 | 575.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 588.30 | 574.26 | 575.69 | SL hit (close>static) qty=1.00 sl=582.45 alert=retest2 |

### Cycle 15 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 600.00 | 579.41 | 577.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 09:15:00 | 633.50 | 600.03 | 594.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 618.45 | 618.68 | 609.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:45:00 | 618.35 | 618.68 | 609.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 625.90 | 631.01 | 626.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 623.45 | 631.01 | 626.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 626.80 | 630.17 | 626.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:15:00 | 628.05 | 630.17 | 626.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 642.40 | 629.53 | 627.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 12:15:00 | 625.60 | 629.99 | 630.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 625.60 | 629.99 | 630.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 623.00 | 627.56 | 628.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 14:15:00 | 622.95 | 621.78 | 623.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 14:15:00 | 622.95 | 621.78 | 623.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 622.95 | 621.78 | 623.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 621.75 | 621.78 | 623.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 623.00 | 622.02 | 623.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 620.80 | 622.02 | 623.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 623.15 | 622.25 | 623.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 624.55 | 622.25 | 623.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 620.65 | 621.93 | 623.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 623.50 | 621.93 | 623.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 606.45 | 604.61 | 610.59 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 612.80 | 607.50 | 607.42 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 604.05 | 607.58 | 607.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 600.65 | 605.15 | 606.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 599.40 | 599.15 | 601.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 599.40 | 599.15 | 601.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 599.40 | 599.15 | 601.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 595.75 | 597.95 | 600.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 565.96 | 575.57 | 582.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 563.40 | 563.36 | 571.56 | SL hit (close>ema200) qty=0.50 sl=563.36 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 585.90 | 573.08 | 572.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 12:15:00 | 597.80 | 590.94 | 585.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 607.40 | 608.16 | 602.71 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 15:00:00 | 612.70 | 609.07 | 603.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 605.70 | 608.57 | 604.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 605.70 | 608.57 | 604.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 600.45 | 606.95 | 604.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 600.45 | 606.95 | 604.00 | SL hit (close<ema400) qty=1.00 sl=604.00 alert=retest1 |

### Cycle 20 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 594.50 | 601.47 | 602.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 589.45 | 599.06 | 600.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 595.55 | 595.52 | 598.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 595.55 | 595.52 | 598.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 600.70 | 596.56 | 598.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:45:00 | 600.40 | 596.56 | 598.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 595.40 | 596.33 | 598.05 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 675.05 | 613.47 | 605.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 678.90 | 651.34 | 628.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 686.10 | 688.93 | 667.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:15:00 | 701.00 | 688.93 | 667.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 805.00 | 796.86 | 789.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:30:00 | 812.25 | 800.17 | 792.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:15:00 | 810.00 | 800.17 | 792.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 777.50 | 793.30 | 792.06 | SL hit (close<static) qty=1.00 sl=785.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 779.15 | 788.93 | 790.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 774.20 | 783.21 | 786.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 779.40 | 776.03 | 780.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 779.40 | 776.03 | 780.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 779.40 | 776.03 | 780.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 778.30 | 776.03 | 780.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 790.75 | 778.97 | 781.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 790.75 | 778.97 | 781.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 784.40 | 780.06 | 781.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:15:00 | 783.05 | 780.06 | 781.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 807.90 | 786.63 | 783.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 807.90 | 786.63 | 783.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 828.80 | 795.06 | 788.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 849.40 | 852.63 | 840.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:30:00 | 849.35 | 852.63 | 840.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 853.00 | 852.71 | 841.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:30:00 | 839.60 | 852.71 | 841.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 853.90 | 852.36 | 846.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 860.65 | 851.59 | 847.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 858.75 | 852.45 | 848.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 12:15:00 | 856.95 | 854.17 | 849.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 845.15 | 855.80 | 854.01 | SL hit (close<static) qty=1.00 sl=846.05 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 14:15:00 | 839.95 | 850.76 | 851.91 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 871.40 | 851.46 | 849.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 852.00 | 855.06 | 855.35 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 861.00 | 856.25 | 855.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 865.95 | 858.93 | 857.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 11:15:00 | 893.35 | 898.13 | 886.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 12:00:00 | 893.35 | 898.13 | 886.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 888.00 | 895.05 | 886.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:45:00 | 886.00 | 895.05 | 886.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 888.50 | 893.74 | 887.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:30:00 | 886.70 | 893.74 | 887.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 888.50 | 892.69 | 887.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 880.55 | 892.69 | 887.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 881.70 | 890.49 | 886.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 882.85 | 890.49 | 886.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 877.00 | 887.80 | 885.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 876.50 | 887.80 | 885.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 876.00 | 883.50 | 884.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 10:15:00 | 874.20 | 878.91 | 881.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 824.00 | 816.32 | 824.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 824.00 | 816.32 | 824.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 824.00 | 816.32 | 824.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 824.00 | 816.32 | 824.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 827.40 | 818.53 | 824.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 829.10 | 818.53 | 824.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 827.00 | 820.23 | 824.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:45:00 | 826.40 | 820.23 | 824.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 833.55 | 827.26 | 827.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:45:00 | 829.30 | 826.89 | 827.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 839.75 | 827.00 | 826.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 839.75 | 827.00 | 826.79 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 815.95 | 829.61 | 830.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 808.85 | 822.30 | 826.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 822.25 | 821.12 | 825.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 822.25 | 821.12 | 825.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 822.25 | 821.12 | 825.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 831.45 | 821.12 | 825.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 811.25 | 815.08 | 819.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 815.20 | 815.08 | 819.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 815.50 | 815.16 | 819.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 815.50 | 815.16 | 819.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 820.20 | 816.44 | 818.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 820.20 | 816.44 | 818.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 836.55 | 820.46 | 820.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 836.55 | 820.46 | 820.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 836.50 | 823.67 | 821.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 847.50 | 832.85 | 827.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 855.05 | 863.80 | 856.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 855.05 | 863.80 | 856.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 855.05 | 863.80 | 856.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 855.05 | 863.80 | 856.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 873.90 | 865.82 | 857.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 878.00 | 868.84 | 859.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 878.05 | 873.47 | 867.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:45:00 | 876.30 | 873.06 | 868.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 884.45 | 872.89 | 868.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 872.35 | 873.74 | 871.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 872.35 | 873.74 | 871.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 884.00 | 875.79 | 872.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:30:00 | 895.35 | 878.65 | 874.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:00:00 | 886.00 | 880.12 | 875.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:30:00 | 886.85 | 883.04 | 877.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 926.60 | 934.47 | 935.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 926.60 | 934.47 | 935.46 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 963.40 | 938.71 | 936.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 979.85 | 946.94 | 940.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 1003.20 | 1005.62 | 995.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 12:00:00 | 1003.20 | 1005.62 | 995.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1003.75 | 1004.89 | 997.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 1000.00 | 1004.89 | 997.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1003.70 | 1004.45 | 999.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 1009.55 | 1004.69 | 1000.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 994.20 | 1002.09 | 1000.32 | SL hit (close<static) qty=1.00 sl=998.15 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 989.00 | 997.54 | 998.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 963.90 | 987.65 | 991.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 979.70 | 973.74 | 980.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 979.70 | 973.74 | 980.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 984.90 | 975.97 | 980.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 984.90 | 975.97 | 980.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 985.20 | 977.82 | 981.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 985.20 | 977.82 | 981.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 987.80 | 979.81 | 981.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 991.60 | 979.81 | 981.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1087.20 | 1001.17 | 991.03 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 15:15:00 | 995.90 | 1020.92 | 1023.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 10:15:00 | 977.80 | 1008.13 | 1016.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 1017.20 | 979.96 | 987.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1017.20 | 979.96 | 987.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1017.20 | 979.96 | 987.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 1030.00 | 979.96 | 987.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1007.30 | 985.43 | 989.40 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 1034.00 | 995.14 | 993.45 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 981.00 | 1021.55 | 1022.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 971.20 | 1011.48 | 1017.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 894.00 | 890.53 | 908.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 894.00 | 890.53 | 908.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 894.00 | 890.53 | 908.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 881.40 | 890.42 | 907.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:45:00 | 882.90 | 888.91 | 900.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 878.00 | 888.91 | 900.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 877.40 | 884.66 | 896.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 885.40 | 884.10 | 890.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 882.90 | 883.86 | 889.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 882.80 | 884.15 | 889.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 881.00 | 883.52 | 888.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 883.20 | 885.68 | 886.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 883.20 | 885.19 | 885.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 888.70 | 885.19 | 885.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 895.10 | 887.17 | 886.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 895.10 | 887.17 | 886.82 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 882.60 | 886.25 | 886.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 877.60 | 884.52 | 885.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 804.80 | 804.42 | 813.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 09:15:00 | 790.00 | 804.42 | 813.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 810.45 | 804.94 | 809.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 810.45 | 804.94 | 809.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 813.40 | 806.63 | 810.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 813.40 | 806.63 | 810.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 813.80 | 808.06 | 810.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 813.00 | 808.06 | 810.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 817.35 | 809.92 | 811.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 817.00 | 809.92 | 811.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 811.50 | 810.24 | 811.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:45:00 | 806.95 | 810.00 | 811.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 807.85 | 810.00 | 811.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 806.80 | 804.93 | 805.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 807.85 | 804.93 | 805.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 807.10 | 805.83 | 805.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 807.10 | 805.83 | 805.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 811.60 | 807.81 | 806.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 804.80 | 811.65 | 810.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 804.80 | 811.65 | 810.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 804.80 | 811.65 | 810.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 805.55 | 811.65 | 810.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 802.35 | 809.79 | 809.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 802.35 | 809.79 | 809.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 800.90 | 808.01 | 808.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 797.90 | 803.48 | 806.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 776.80 | 774.59 | 786.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 09:30:00 | 770.45 | 774.59 | 786.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 792.05 | 779.25 | 786.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:30:00 | 792.80 | 779.25 | 786.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 807.20 | 784.84 | 788.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 807.20 | 784.84 | 788.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 823.95 | 796.93 | 793.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 839.30 | 817.18 | 806.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 845.90 | 846.99 | 832.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 857.50 | 846.99 | 832.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 909.50 | 919.82 | 905.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:30:00 | 924.35 | 919.94 | 909.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:30:00 | 925.65 | 926.35 | 916.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 924.85 | 925.40 | 917.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 14:45:00 | 931.65 | 924.85 | 918.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 921.10 | 924.04 | 919.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 923.90 | 924.04 | 919.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 920.65 | 923.36 | 919.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 933.85 | 920.34 | 919.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 929.05 | 947.85 | 949.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 929.05 | 947.85 | 949.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 924.80 | 943.24 | 947.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 850.80 | 849.58 | 863.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:15:00 | 853.60 | 849.58 | 863.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 869.00 | 853.00 | 862.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 869.00 | 853.00 | 862.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 878.45 | 858.09 | 863.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 15:00:00 | 878.45 | 858.09 | 863.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 727.00 | 720.06 | 730.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 730.80 | 720.06 | 730.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 710.80 | 711.43 | 719.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 712.70 | 711.43 | 719.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 717.85 | 708.85 | 713.35 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 731.50 | 718.37 | 716.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 735.40 | 721.77 | 718.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 781.10 | 786.12 | 769.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 781.10 | 786.12 | 769.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 779.15 | 784.73 | 770.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 771.60 | 784.73 | 770.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 763.50 | 779.40 | 770.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 763.50 | 779.40 | 770.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 757.00 | 774.92 | 769.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 766.55 | 772.51 | 768.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 746.90 | 767.39 | 766.71 | SL hit (close<static) qty=1.00 sl=752.10 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 755.75 | 765.06 | 765.72 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 776.80 | 766.58 | 765.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 777.50 | 770.39 | 767.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 788.10 | 793.73 | 787.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 788.10 | 793.73 | 787.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 788.10 | 793.73 | 787.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 787.00 | 793.73 | 787.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 781.00 | 791.19 | 786.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 781.00 | 791.19 | 786.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 781.55 | 789.26 | 786.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 780.10 | 789.26 | 786.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 778.00 | 783.57 | 784.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 766.45 | 780.15 | 782.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 784.90 | 780.60 | 782.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 784.90 | 780.60 | 782.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 784.90 | 780.60 | 782.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 784.90 | 780.60 | 782.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 779.30 | 780.34 | 781.97 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 786.00 | 783.31 | 783.10 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 09:15:00 | 769.50 | 780.55 | 781.86 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 804.60 | 784.95 | 782.65 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 771.90 | 781.48 | 782.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 762.80 | 775.19 | 778.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 751.00 | 747.39 | 757.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:15:00 | 752.85 | 747.39 | 757.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 740.40 | 745.82 | 751.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 738.00 | 745.82 | 751.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:15:00 | 737.00 | 744.48 | 750.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 737.85 | 737.77 | 742.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 14:15:00 | 701.10 | 711.77 | 720.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 700.15 | 708.05 | 717.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 700.96 | 708.05 | 717.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 704.00 | 702.77 | 710.10 | SL hit (close>ema200) qty=0.50 sl=702.77 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 682.35 | 666.39 | 665.58 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 651.25 | 667.08 | 667.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 646.10 | 657.89 | 663.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 662.65 | 657.16 | 661.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 662.65 | 657.16 | 661.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 662.65 | 657.16 | 661.28 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 676.65 | 664.74 | 663.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 680.65 | 667.92 | 665.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 676.05 | 677.55 | 672.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 676.05 | 677.55 | 672.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 671.65 | 676.21 | 672.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 671.65 | 676.21 | 672.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 670.50 | 675.07 | 672.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 663.75 | 675.07 | 672.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 682.75 | 674.65 | 672.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 689.15 | 676.74 | 673.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 668.00 | 679.65 | 676.53 | SL hit (close<static) qty=1.00 sl=672.05 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 664.30 | 674.23 | 674.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 659.85 | 671.36 | 673.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 658.30 | 653.54 | 660.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 658.30 | 653.54 | 660.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 658.30 | 653.54 | 660.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 661.70 | 653.54 | 660.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 666.25 | 656.49 | 660.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 666.25 | 656.49 | 660.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 660.55 | 657.31 | 660.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 659.25 | 657.31 | 660.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 659.15 | 657.87 | 660.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 684.90 | 666.42 | 663.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 684.90 | 666.42 | 663.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 692.20 | 671.57 | 666.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 667.05 | 675.93 | 671.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 667.05 | 675.93 | 671.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 667.05 | 675.93 | 671.68 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 660.00 | 670.07 | 670.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 659.00 | 667.86 | 669.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 671.05 | 667.40 | 668.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 671.05 | 667.40 | 668.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 671.05 | 667.40 | 668.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 674.00 | 667.40 | 668.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 672.75 | 668.47 | 668.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 673.65 | 668.47 | 668.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 669.00 | 668.58 | 668.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 674.80 | 668.58 | 668.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 664.85 | 667.83 | 668.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 661.05 | 665.30 | 667.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 658.80 | 665.30 | 667.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 642.95 | 664.62 | 666.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:15:00 | 628.00 | 647.26 | 657.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 625.86 | 641.88 | 654.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 650.25 | 637.65 | 647.59 | SL hit (close>ema200) qty=0.50 sl=637.65 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 659.30 | 651.74 | 651.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 674.05 | 657.54 | 654.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 658.25 | 665.53 | 661.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 658.25 | 665.53 | 661.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 658.25 | 665.53 | 661.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 658.25 | 665.53 | 661.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 653.00 | 663.02 | 660.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 653.00 | 663.02 | 660.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 652.00 | 657.86 | 658.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 642.00 | 654.69 | 657.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 659.95 | 633.52 | 640.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 659.95 | 633.52 | 640.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 659.95 | 633.52 | 640.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 659.95 | 633.52 | 640.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 658.10 | 638.44 | 642.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 659.50 | 638.44 | 642.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 669.10 | 648.70 | 646.70 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 632.85 | 646.98 | 647.33 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 661.70 | 648.81 | 647.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 668.30 | 652.70 | 649.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 777.70 | 779.51 | 764.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 12:00:00 | 777.70 | 779.51 | 764.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 783.00 | 785.05 | 778.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 796.30 | 783.65 | 781.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:00:00 | 798.60 | 807.98 | 805.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 792.05 | 802.56 | 803.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 792.05 | 802.56 | 803.56 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 811.45 | 804.43 | 803.99 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 799.60 | 805.55 | 805.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 797.80 | 804.00 | 805.04 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 825.60 | 807.37 | 806.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 828.00 | 811.50 | 808.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 810.35 | 812.40 | 809.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 810.35 | 812.40 | 809.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 808.20 | 811.56 | 809.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 807.40 | 811.56 | 809.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 792.90 | 807.83 | 807.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 792.90 | 807.83 | 807.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 793.70 | 805.00 | 806.28 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 807.65 | 800.37 | 799.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 841.90 | 808.67 | 803.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 848.95 | 852.01 | 837.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 848.95 | 852.01 | 837.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-27 14:00:00 | 561.70 | 2025-05-28 09:15:00 | 572.20 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-06-04 12:00:00 | 609.95 | 2025-06-09 14:15:00 | 603.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-04 14:45:00 | 608.45 | 2025-06-09 14:15:00 | 603.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-04 15:15:00 | 608.90 | 2025-06-09 14:15:00 | 603.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-06-06 09:15:00 | 609.00 | 2025-06-09 14:15:00 | 603.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-06 11:15:00 | 608.80 | 2025-06-09 15:15:00 | 604.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-06-09 09:15:00 | 608.80 | 2025-06-09 15:15:00 | 604.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-06-09 11:15:00 | 608.75 | 2025-06-09 15:15:00 | 604.20 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-09 12:15:00 | 608.80 | 2025-06-09 15:15:00 | 604.20 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-06-20 12:15:00 | 575.20 | 2025-06-24 09:15:00 | 588.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-06-20 13:00:00 | 576.20 | 2025-06-24 09:15:00 | 588.30 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-06-20 13:30:00 | 576.70 | 2025-06-24 09:15:00 | 588.30 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-06-23 09:15:00 | 574.00 | 2025-06-24 09:15:00 | 588.30 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-07-04 13:15:00 | 628.05 | 2025-07-08 12:15:00 | 625.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-07 09:15:00 | 642.40 | 2025-07-08 12:15:00 | 625.60 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-07-22 14:00:00 | 595.75 | 2025-07-28 09:15:00 | 565.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:00:00 | 595.75 | 2025-07-29 09:15:00 | 563.40 | STOP_HIT | 0.50 | 5.43% |
| BUY | retest1 | 2025-08-05 15:00:00 | 612.70 | 2025-08-06 10:15:00 | 600.45 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-22 12:30:00 | 812.25 | 2025-08-25 10:15:00 | 777.50 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-08-22 13:15:00 | 810.00 | 2025-08-25 10:15:00 | 777.50 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2025-08-28 13:15:00 | 783.05 | 2025-08-29 10:15:00 | 807.90 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-09-05 09:15:00 | 860.65 | 2025-09-08 12:15:00 | 845.15 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-09-05 10:15:00 | 858.75 | 2025-09-08 12:15:00 | 845.15 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-05 12:15:00 | 856.95 | 2025-09-08 12:15:00 | 845.15 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-25 11:45:00 | 829.30 | 2025-09-26 09:15:00 | 839.75 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-08 12:45:00 | 878.00 | 2025-10-24 09:15:00 | 926.60 | STOP_HIT | 1.00 | 5.54% |
| BUY | retest2 | 2025-10-09 13:15:00 | 878.05 | 2025-10-24 09:15:00 | 926.60 | STOP_HIT | 1.00 | 5.53% |
| BUY | retest2 | 2025-10-09 14:45:00 | 876.30 | 2025-10-24 09:15:00 | 926.60 | STOP_HIT | 1.00 | 5.74% |
| BUY | retest2 | 2025-10-10 09:15:00 | 884.45 | 2025-10-24 09:15:00 | 926.60 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-10-13 09:30:00 | 895.35 | 2025-10-24 09:15:00 | 926.60 | STOP_HIT | 1.00 | 3.49% |
| BUY | retest2 | 2025-10-13 11:00:00 | 886.00 | 2025-10-24 09:15:00 | 926.60 | STOP_HIT | 1.00 | 4.58% |
| BUY | retest2 | 2025-10-13 11:30:00 | 886.85 | 2025-10-24 09:15:00 | 926.60 | STOP_HIT | 1.00 | 4.48% |
| BUY | retest2 | 2025-10-31 14:00:00 | 1009.55 | 2025-11-03 09:15:00 | 994.20 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-11-25 10:45:00 | 881.40 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-25 14:45:00 | 882.90 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-25 15:15:00 | 878.00 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-26 09:30:00 | 877.40 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-11-27 11:00:00 | 882.90 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-27 12:15:00 | 882.80 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-27 13:00:00 | 881.00 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-28 15:15:00 | 883.20 | 2025-12-01 09:15:00 | 895.10 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-10 11:45:00 | 806.95 | 2025-12-12 12:15:00 | 807.10 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-12-10 12:15:00 | 807.85 | 2025-12-12 12:15:00 | 807.10 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-12-12 10:45:00 | 806.80 | 2025-12-12 12:15:00 | 807.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-12-12 11:15:00 | 807.85 | 2025-12-12 12:15:00 | 807.10 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-30 13:30:00 | 924.35 | 2026-01-08 10:15:00 | 929.05 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-12-31 10:30:00 | 925.65 | 2026-01-08 10:15:00 | 929.05 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-12-31 12:45:00 | 924.85 | 2026-01-08 10:15:00 | 929.05 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-12-31 14:45:00 | 931.65 | 2026-01-08 10:15:00 | 929.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-01-02 09:15:00 | 933.85 | 2026-01-08 10:15:00 | 929.05 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-02 09:45:00 | 766.55 | 2026-02-02 10:15:00 | 746.90 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-02-18 10:15:00 | 738.00 | 2026-02-23 14:15:00 | 701.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 11:15:00 | 737.00 | 2026-02-24 09:15:00 | 700.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:45:00 | 737.85 | 2026-02-24 09:15:00 | 700.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 738.00 | 2026-02-24 15:15:00 | 704.00 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2026-02-18 11:15:00 | 737.00 | 2026-02-24 15:15:00 | 704.00 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2026-02-19 11:45:00 | 737.85 | 2026-02-24 15:15:00 | 704.00 | STOP_HIT | 0.50 | 4.59% |
| BUY | retest2 | 2026-03-12 13:15:00 | 689.15 | 2026-03-13 09:15:00 | 668.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-03-17 11:15:00 | 659.25 | 2026-03-18 09:15:00 | 684.90 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2026-03-17 12:15:00 | 659.15 | 2026-03-18 09:15:00 | 684.90 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-03-20 14:30:00 | 661.05 | 2026-03-23 11:15:00 | 628.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 15:00:00 | 658.80 | 2026-03-23 12:15:00 | 625.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 661.05 | 2026-03-24 09:15:00 | 650.25 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-03-20 15:00:00 | 658.80 | 2026-03-24 09:15:00 | 650.25 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2026-03-23 09:15:00 | 642.95 | 2026-03-24 14:15:00 | 659.30 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-03-24 13:45:00 | 661.55 | 2026-03-24 14:15:00 | 659.30 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2026-04-22 09:15:00 | 796.30 | 2026-04-24 13:15:00 | 792.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-04-24 12:00:00 | 798.60 | 2026-04-24 13:15:00 | 792.05 | STOP_HIT | 1.00 | -0.82% |
