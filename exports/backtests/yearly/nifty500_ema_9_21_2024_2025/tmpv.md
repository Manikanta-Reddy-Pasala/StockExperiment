# Tata Motors Passenger Vehicles Ltd. (TMPV)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 355.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 102 |
| ALERT2 | 102 |
| ALERT2_SKIP | 48 |
| ALERT3 | 232 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 118 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 92
- **Target hits / Stop hits / Partials:** 4 / 114 / 14
- **Avg / median % per leg:** 0.21% / -0.72%
- **Sum % (uncompounded):** 27.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 11 | 20.8% | 2 | 51 | 0 | -0.17% | -9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 53 | 11 | 20.8% | 2 | 51 | 0 | -0.17% | -9.2% |
| SELL (all) | 79 | 29 | 36.7% | 2 | 63 | 14 | 0.46% | 36.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 79 | 29 | 36.7% | 2 | 63 | 14 | 0.46% | 36.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 132 | 40 | 30.3% | 4 | 114 | 14 | 0.21% | 27.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 578.76 | 619.66 | 620.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 14:15:00 | 574.30 | 579.19 | 586.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 567.97 | 567.71 | 576.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 567.97 | 567.71 | 576.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 571.27 | 568.66 | 575.16 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 10:15:00 | 578.58 | 575.70 | 575.48 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 573.94 | 575.55 | 575.74 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 13:15:00 | 582.67 | 577.06 | 576.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 583.61 | 578.37 | 576.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 582.06 | 583.55 | 580.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 582.06 | 583.55 | 580.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 581.15 | 583.07 | 580.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 584.24 | 583.07 | 580.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 580.15 | 582.49 | 580.85 | SL hit (close<static) qty=1.00 sl=580.61 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 575.09 | 580.04 | 580.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 574.36 | 577.70 | 579.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 574.39 | 574.33 | 576.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:45:00 | 574.12 | 574.33 | 576.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 568.64 | 572.42 | 575.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:00:00 | 568.36 | 571.61 | 574.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 566.94 | 565.27 | 566.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 12:15:00 | 574.67 | 568.44 | 568.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 574.67 | 568.44 | 568.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 576.33 | 570.02 | 568.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 564.21 | 570.72 | 569.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 564.21 | 570.72 | 569.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 564.21 | 570.72 | 569.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 558.48 | 570.72 | 569.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 545.48 | 565.67 | 567.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 524.55 | 557.45 | 563.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 557.55 | 553.38 | 558.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 557.55 | 553.38 | 558.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 557.55 | 553.38 | 558.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 557.55 | 553.38 | 558.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 557.24 | 554.15 | 558.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 556.91 | 554.15 | 558.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 559.82 | 555.28 | 558.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 559.82 | 555.28 | 558.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 562.42 | 556.71 | 558.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 564.00 | 556.71 | 558.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 569.15 | 561.14 | 560.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 572.76 | 563.47 | 561.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 14:15:00 | 599.27 | 600.11 | 595.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 599.27 | 600.11 | 595.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 598.70 | 599.22 | 596.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 599.48 | 599.22 | 596.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:15:00 | 600.06 | 597.98 | 596.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:00:00 | 599.82 | 598.35 | 596.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 12:00:00 | 599.42 | 600.66 | 599.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 599.36 | 600.40 | 599.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:30:00 | 598.67 | 600.40 | 599.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 597.58 | 599.84 | 599.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:00:00 | 597.58 | 599.84 | 599.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 597.52 | 599.37 | 598.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:30:00 | 597.00 | 599.37 | 598.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 595.76 | 598.41 | 598.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 595.76 | 598.41 | 598.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 595.06 | 597.09 | 597.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 595.82 | 595.15 | 596.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 595.82 | 595.15 | 596.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 595.82 | 595.15 | 596.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 12:45:00 | 592.27 | 594.94 | 596.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 587.97 | 594.21 | 595.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 590.39 | 580.73 | 580.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 590.39 | 580.73 | 580.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 597.06 | 585.39 | 582.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 10:15:00 | 597.58 | 597.83 | 592.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 11:00:00 | 597.58 | 597.83 | 592.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 599.30 | 602.25 | 597.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 599.88 | 602.25 | 597.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 595.70 | 600.69 | 597.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 595.70 | 600.69 | 597.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 595.52 | 599.66 | 597.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:30:00 | 595.64 | 599.66 | 597.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 595.88 | 597.72 | 596.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 593.94 | 597.72 | 596.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 10:15:00 | 591.18 | 595.48 | 596.00 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 597.70 | 595.85 | 595.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 13:15:00 | 600.76 | 597.24 | 596.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 14:15:00 | 602.61 | 602.93 | 600.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 14:15:00 | 602.61 | 602.93 | 600.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 602.61 | 602.93 | 600.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 600.79 | 602.93 | 600.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 601.79 | 602.71 | 600.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 613.21 | 602.71 | 600.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 612.27 | 618.03 | 618.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 612.27 | 618.03 | 618.51 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 14:15:00 | 621.30 | 619.02 | 618.81 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 613.48 | 618.26 | 618.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 611.91 | 616.99 | 617.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 608.48 | 606.15 | 609.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 14:15:00 | 608.48 | 606.15 | 609.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 608.48 | 606.15 | 609.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 609.85 | 606.15 | 609.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 608.42 | 606.60 | 609.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 608.79 | 606.60 | 609.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 603.88 | 606.06 | 608.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 591.45 | 606.72 | 608.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 603.24 | 606.30 | 608.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 623.48 | 609.83 | 609.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 623.48 | 609.83 | 609.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 644.85 | 623.58 | 617.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 10:15:00 | 696.00 | 700.53 | 694.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 10:15:00 | 696.00 | 700.53 | 694.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 696.00 | 700.53 | 694.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 696.00 | 700.53 | 694.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 693.91 | 699.21 | 694.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 694.12 | 699.21 | 694.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 693.88 | 698.14 | 694.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 692.91 | 698.14 | 694.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 689.24 | 696.36 | 693.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 690.55 | 696.36 | 693.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 694.12 | 695.91 | 693.64 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 669.64 | 690.08 | 691.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 637.33 | 665.31 | 676.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 14:15:00 | 621.24 | 619.16 | 628.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 15:00:00 | 621.24 | 619.16 | 628.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 633.18 | 622.45 | 628.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:30:00 | 632.48 | 622.45 | 628.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 631.00 | 624.16 | 628.50 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 650.45 | 634.08 | 632.08 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 637.30 | 642.94 | 643.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 635.52 | 640.69 | 642.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 640.85 | 640.72 | 642.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 640.85 | 640.72 | 642.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 641.55 | 640.89 | 642.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:45:00 | 642.06 | 640.89 | 642.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 643.82 | 641.47 | 642.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:30:00 | 643.30 | 641.47 | 642.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 643.39 | 641.86 | 642.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:15:00 | 644.18 | 641.86 | 642.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 644.18 | 642.32 | 642.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 659.52 | 642.32 | 642.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 657.45 | 645.35 | 643.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 665.36 | 656.00 | 650.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 658.91 | 660.18 | 655.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 15:00:00 | 658.91 | 660.18 | 655.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 657.48 | 659.70 | 656.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 657.48 | 659.70 | 656.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 657.76 | 659.31 | 656.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 657.76 | 659.31 | 656.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 658.24 | 659.10 | 656.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 657.18 | 659.10 | 656.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 660.61 | 659.08 | 657.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 659.27 | 659.08 | 657.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 658.15 | 658.76 | 657.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:00:00 | 658.15 | 658.76 | 657.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 658.06 | 658.56 | 657.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:30:00 | 657.67 | 658.56 | 657.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 657.39 | 658.32 | 657.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:00:00 | 657.39 | 658.32 | 657.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 658.70 | 658.40 | 657.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:15:00 | 655.33 | 658.40 | 657.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 09:15:00 | 653.09 | 657.34 | 657.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 649.97 | 654.43 | 655.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 654.64 | 652.14 | 654.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 654.64 | 652.14 | 654.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 654.64 | 652.14 | 654.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 654.15 | 652.14 | 654.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 659.00 | 653.51 | 654.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:45:00 | 658.30 | 653.51 | 654.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 659.45 | 654.70 | 654.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 659.67 | 654.70 | 654.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 12:15:00 | 659.88 | 655.73 | 655.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 663.70 | 658.10 | 656.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 09:15:00 | 656.94 | 661.01 | 659.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 656.94 | 661.01 | 659.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 656.94 | 661.01 | 659.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 656.94 | 661.01 | 659.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 656.64 | 660.14 | 659.22 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 654.48 | 657.94 | 658.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 653.21 | 656.99 | 657.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 655.36 | 653.44 | 654.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 655.36 | 653.44 | 654.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 655.36 | 653.44 | 654.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 655.36 | 653.44 | 654.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 651.45 | 653.04 | 654.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 11:45:00 | 649.24 | 651.87 | 653.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 14:15:00 | 680.03 | 657.03 | 655.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 14:15:00 | 680.03 | 657.03 | 655.68 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 660.52 | 662.65 | 662.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 658.55 | 661.13 | 662.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 655.27 | 655.06 | 657.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 15:00:00 | 655.27 | 655.06 | 657.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 654.30 | 654.92 | 657.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:15:00 | 650.94 | 654.52 | 656.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:00:00 | 650.70 | 652.83 | 655.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:15:00 | 618.39 | 622.18 | 629.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:15:00 | 618.16 | 622.18 | 629.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-12 09:15:00 | 585.85 | 597.35 | 610.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 588.73 | 588.08 | 588.05 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 11:15:00 | 586.48 | 587.76 | 587.91 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 588.52 | 588.03 | 588.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 592.73 | 589.29 | 588.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 584.27 | 590.14 | 589.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 584.27 | 590.14 | 589.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 584.27 | 590.14 | 589.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 584.27 | 590.14 | 589.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 582.91 | 588.69 | 589.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 581.55 | 587.27 | 588.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 591.82 | 586.51 | 587.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 591.82 | 586.51 | 587.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 591.82 | 586.51 | 587.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 591.45 | 586.51 | 587.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 594.55 | 588.11 | 588.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 13:15:00 | 598.06 | 592.66 | 590.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 15:15:00 | 599.94 | 600.13 | 596.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 09:15:00 | 595.27 | 600.13 | 596.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 595.15 | 599.14 | 596.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 595.18 | 599.14 | 596.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 595.30 | 598.37 | 596.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 594.09 | 598.37 | 596.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 591.58 | 595.45 | 595.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 12:15:00 | 590.42 | 593.25 | 594.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 573.15 | 569.47 | 576.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 573.15 | 569.47 | 576.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 564.48 | 566.21 | 571.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:45:00 | 560.70 | 565.22 | 570.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 12:45:00 | 561.00 | 563.93 | 569.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 14:45:00 | 562.52 | 562.95 | 567.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 544.85 | 562.88 | 567.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 569.73 | 559.77 | 562.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 569.73 | 559.77 | 562.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 572.33 | 562.28 | 563.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 572.33 | 562.28 | 563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 570.97 | 564.02 | 563.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 570.97 | 564.02 | 563.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 573.36 | 567.16 | 565.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 565.64 | 567.72 | 566.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 10:15:00 | 565.64 | 567.72 | 566.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 565.64 | 567.72 | 566.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 565.64 | 567.72 | 566.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 564.73 | 567.12 | 566.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 563.03 | 567.12 | 566.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 562.39 | 565.29 | 565.46 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 568.91 | 565.49 | 565.35 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 13:15:00 | 563.30 | 565.19 | 565.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 09:15:00 | 559.27 | 563.34 | 564.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 547.61 | 545.45 | 549.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 09:45:00 | 546.73 | 545.45 | 549.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 550.79 | 546.51 | 549.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:30:00 | 550.91 | 546.51 | 549.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 550.70 | 547.35 | 549.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:45:00 | 551.64 | 547.35 | 549.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 554.21 | 548.72 | 550.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:15:00 | 554.76 | 548.72 | 550.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 549.58 | 549.87 | 550.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 547.94 | 549.30 | 550.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 547.82 | 548.44 | 549.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 520.54 | 530.59 | 534.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 520.43 | 530.59 | 534.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 526.88 | 525.73 | 529.59 | SL hit (close>ema200) qty=0.50 sl=525.73 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 534.45 | 531.82 | 531.53 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 516.55 | 528.76 | 530.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 510.06 | 525.02 | 528.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 513.09 | 508.67 | 511.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 513.09 | 508.67 | 511.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 513.09 | 508.67 | 511.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 513.09 | 508.67 | 511.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 510.91 | 509.11 | 511.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 502.33 | 509.11 | 511.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 12:15:00 | 508.33 | 505.76 | 505.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 508.33 | 505.76 | 505.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 508.85 | 506.38 | 505.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 504.85 | 506.85 | 506.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 504.85 | 506.85 | 506.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 504.85 | 506.85 | 506.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 504.85 | 506.85 | 506.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 502.85 | 506.05 | 505.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 500.30 | 506.05 | 505.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 500.70 | 504.98 | 505.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 12:15:00 | 497.52 | 503.49 | 504.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 495.70 | 491.89 | 496.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 495.70 | 491.89 | 496.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 495.70 | 491.89 | 496.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:00:00 | 495.70 | 491.89 | 496.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 501.88 | 493.89 | 496.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 501.88 | 493.89 | 496.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 497.39 | 494.59 | 496.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 494.58 | 495.18 | 496.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 12:15:00 | 469.85 | 474.39 | 478.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 469.88 | 469.62 | 474.05 | SL hit (close>ema200) qty=0.50 sl=469.62 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 480.58 | 474.22 | 474.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 483.15 | 476.01 | 474.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 473.64 | 476.68 | 475.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 473.64 | 476.68 | 475.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 473.64 | 476.68 | 475.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 473.64 | 476.68 | 475.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 473.91 | 476.12 | 475.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 467.21 | 476.12 | 475.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 467.73 | 474.45 | 474.69 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 478.97 | 473.99 | 473.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 479.48 | 475.09 | 474.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 478.55 | 481.62 | 479.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 11:15:00 | 478.55 | 481.62 | 479.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 478.55 | 481.62 | 479.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:00:00 | 478.55 | 481.62 | 479.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 476.36 | 480.57 | 479.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:00:00 | 476.36 | 480.57 | 479.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 474.67 | 478.46 | 478.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 472.12 | 475.65 | 476.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 474.94 | 474.81 | 475.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:30:00 | 474.39 | 474.81 | 475.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 474.67 | 474.78 | 475.77 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 481.52 | 476.80 | 476.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 12:15:00 | 482.88 | 480.63 | 478.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 479.55 | 482.67 | 480.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 10:15:00 | 479.55 | 482.67 | 480.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 479.55 | 482.67 | 480.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 479.55 | 482.67 | 480.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 477.42 | 481.62 | 480.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 477.42 | 481.62 | 480.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 15:15:00 | 477.70 | 479.63 | 479.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 475.15 | 478.73 | 479.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 478.15 | 477.98 | 478.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 12:00:00 | 478.15 | 477.98 | 478.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 477.45 | 477.88 | 478.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 13:15:00 | 476.42 | 477.88 | 478.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 13:15:00 | 481.36 | 478.57 | 479.02 | SL hit (close>static) qty=1.00 sl=479.27 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 480.61 | 479.39 | 479.34 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 476.64 | 478.84 | 479.09 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 488.61 | 481.00 | 480.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 14:15:00 | 495.55 | 483.91 | 481.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 13:15:00 | 486.18 | 487.70 | 484.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 14:00:00 | 486.18 | 487.70 | 484.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 484.33 | 487.02 | 484.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 484.33 | 487.02 | 484.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 484.73 | 486.56 | 484.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 488.48 | 486.56 | 484.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 12:45:00 | 485.70 | 486.85 | 485.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 15:15:00 | 485.45 | 485.98 | 485.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 479.52 | 484.44 | 485.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 479.52 | 484.44 | 485.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 476.79 | 481.89 | 483.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 477.91 | 476.91 | 479.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:45:00 | 478.06 | 476.91 | 479.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 479.09 | 477.35 | 479.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 479.09 | 477.35 | 479.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 479.27 | 477.73 | 479.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 477.67 | 477.73 | 479.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 477.67 | 477.72 | 479.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 476.67 | 477.72 | 479.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 476.45 | 477.29 | 478.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:30:00 | 476.00 | 476.65 | 478.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 476.06 | 476.67 | 477.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 478.06 | 476.95 | 477.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 478.39 | 476.95 | 477.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 475.30 | 476.62 | 477.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:15:00 | 474.91 | 476.61 | 477.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 474.24 | 476.34 | 477.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 452.84 | 460.40 | 466.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 452.63 | 460.40 | 466.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 452.20 | 460.40 | 466.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 452.26 | 460.40 | 466.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 451.16 | 460.40 | 466.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 450.53 | 460.40 | 466.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 446.06 | 441.18 | 445.82 | SL hit (close>ema200) qty=0.50 sl=441.18 alert=retest2 |

### Cycle 50 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 450.64 | 447.45 | 447.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 12:15:00 | 451.82 | 448.32 | 447.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 454.97 | 455.33 | 452.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 15:00:00 | 454.97 | 455.33 | 452.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 451.52 | 454.83 | 452.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 451.52 | 454.83 | 452.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 451.42 | 454.15 | 452.58 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 443.52 | 450.19 | 451.08 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 454.15 | 449.85 | 449.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 455.67 | 452.82 | 451.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 472.09 | 475.10 | 468.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 472.09 | 475.10 | 468.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 469.85 | 473.99 | 469.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 464.03 | 473.99 | 469.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 470.06 | 473.20 | 469.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 471.52 | 473.20 | 469.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:00:00 | 473.03 | 473.17 | 470.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 11:00:00 | 472.36 | 477.68 | 476.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 14:15:00 | 472.30 | 475.33 | 475.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 14:15:00 | 472.30 | 475.33 | 475.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 468.94 | 473.77 | 474.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 10:15:00 | 476.52 | 474.32 | 474.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 10:15:00 | 476.52 | 474.32 | 474.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 476.52 | 474.32 | 474.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:00:00 | 476.52 | 474.32 | 474.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 473.48 | 474.15 | 474.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:00:00 | 470.48 | 473.43 | 474.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 471.94 | 467.61 | 467.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 471.94 | 467.61 | 467.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 12:15:00 | 473.15 | 471.11 | 469.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 467.18 | 470.93 | 469.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 467.18 | 470.93 | 469.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 467.18 | 470.93 | 469.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 466.36 | 470.93 | 469.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 468.58 | 470.46 | 469.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:30:00 | 471.91 | 470.42 | 469.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:00:00 | 471.73 | 470.69 | 470.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 474.09 | 470.21 | 469.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 467.61 | 469.83 | 469.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 467.61 | 469.83 | 469.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 460.76 | 467.07 | 468.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 455.15 | 453.30 | 458.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 455.15 | 453.30 | 458.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 458.36 | 455.51 | 458.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 458.30 | 455.51 | 458.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 458.79 | 456.17 | 458.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 457.30 | 456.17 | 458.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 434.44 | 444.15 | 449.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 10:15:00 | 436.58 | 435.78 | 441.85 | SL hit (close>ema200) qty=0.50 sl=435.78 alert=retest2 |

### Cycle 56 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 453.42 | 445.51 | 444.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 456.67 | 449.55 | 446.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 09:15:00 | 422.91 | 445.49 | 445.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 422.91 | 445.49 | 445.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 422.91 | 445.49 | 445.48 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 10:15:00 | 427.30 | 441.85 | 443.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 14:15:00 | 422.36 | 431.46 | 437.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 12:15:00 | 430.61 | 428.79 | 433.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 13:00:00 | 430.61 | 428.79 | 433.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 433.94 | 429.96 | 433.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 433.94 | 429.96 | 433.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 433.76 | 430.72 | 433.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:15:00 | 432.82 | 430.72 | 433.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 434.48 | 431.47 | 433.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 434.30 | 431.47 | 433.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 433.27 | 431.83 | 433.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:30:00 | 434.45 | 431.83 | 433.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 430.85 | 431.63 | 433.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:15:00 | 432.03 | 431.63 | 433.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 428.94 | 431.10 | 432.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 420.52 | 429.97 | 431.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 14:15:00 | 430.36 | 426.94 | 426.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 430.36 | 426.94 | 426.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 436.55 | 429.37 | 428.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 433.00 | 433.74 | 431.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 433.00 | 433.74 | 431.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 432.61 | 433.55 | 431.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:30:00 | 431.79 | 433.55 | 431.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 430.48 | 432.94 | 431.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 430.48 | 432.94 | 431.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 429.76 | 432.30 | 431.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 429.67 | 432.30 | 431.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 433.36 | 431.93 | 431.59 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 428.36 | 430.86 | 431.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 426.91 | 430.07 | 430.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 413.94 | 412.86 | 417.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 413.94 | 412.86 | 417.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 417.64 | 414.71 | 416.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 410.88 | 413.94 | 415.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:15:00 | 411.24 | 411.19 | 413.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 416.33 | 414.30 | 414.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 416.33 | 414.30 | 414.03 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 411.27 | 413.70 | 413.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 10:15:00 | 409.82 | 412.92 | 413.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 14:15:00 | 414.00 | 411.93 | 412.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 14:15:00 | 414.00 | 411.93 | 412.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 414.00 | 411.93 | 412.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 414.00 | 411.93 | 412.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 413.64 | 412.27 | 412.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 414.39 | 412.27 | 412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 414.85 | 413.37 | 413.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 418.97 | 414.69 | 413.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 409.27 | 415.36 | 414.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 409.27 | 415.36 | 414.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 409.27 | 415.36 | 414.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 409.27 | 415.36 | 414.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 409.82 | 414.25 | 414.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 407.52 | 410.24 | 412.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 379.82 | 377.89 | 384.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 379.82 | 377.89 | 384.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 383.30 | 377.59 | 379.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 382.73 | 377.59 | 379.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 385.58 | 379.19 | 380.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 385.58 | 379.19 | 380.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 384.24 | 381.41 | 381.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 388.18 | 383.41 | 382.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 15:15:00 | 391.94 | 392.67 | 390.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 09:15:00 | 392.58 | 392.67 | 390.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 390.85 | 392.30 | 390.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 402.30 | 392.53 | 391.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 406.42 | 425.64 | 427.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 406.42 | 425.64 | 427.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 09:15:00 | 400.30 | 405.68 | 408.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 357.27 | 354.28 | 366.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 357.27 | 354.28 | 366.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 357.27 | 354.28 | 366.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 355.39 | 355.04 | 365.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 355.21 | 355.04 | 365.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 356.36 | 356.94 | 362.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 356.06 | 356.76 | 362.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 365.33 | 357.56 | 359.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 363.45 | 357.56 | 359.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 12:30:00 | 362.97 | 360.66 | 360.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 378.18 | 364.21 | 362.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 378.18 | 364.21 | 362.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 11:15:00 | 384.79 | 378.71 | 375.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 398.97 | 401.40 | 396.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 398.97 | 401.40 | 396.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 398.97 | 401.40 | 396.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 398.97 | 401.40 | 396.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 397.09 | 400.54 | 396.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 396.97 | 400.54 | 396.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 397.55 | 399.94 | 396.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:30:00 | 398.45 | 399.93 | 396.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 09:15:00 | 398.48 | 398.69 | 396.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 390.91 | 401.38 | 401.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 390.91 | 401.38 | 401.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 388.85 | 394.70 | 397.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 398.18 | 394.07 | 396.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 398.18 | 394.07 | 396.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 398.18 | 394.07 | 396.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 398.18 | 394.07 | 396.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 397.79 | 394.81 | 396.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 395.52 | 394.81 | 396.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 401.30 | 396.87 | 396.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 401.30 | 396.87 | 396.76 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 393.55 | 397.33 | 397.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 392.27 | 396.31 | 397.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 403.18 | 396.68 | 396.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 403.18 | 396.68 | 396.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 403.18 | 396.68 | 396.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 406.58 | 396.68 | 396.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 407.45 | 398.84 | 397.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 407.97 | 400.66 | 398.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 414.15 | 414.76 | 409.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 414.15 | 414.76 | 409.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 425.48 | 416.49 | 411.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:00:00 | 428.79 | 422.61 | 416.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:00:00 | 429.39 | 423.96 | 417.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 428.12 | 429.88 | 428.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 11:15:00 | 422.27 | 427.50 | 427.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 422.27 | 427.50 | 427.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 420.21 | 426.04 | 426.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 431.24 | 426.17 | 426.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 431.24 | 426.17 | 426.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 431.24 | 426.17 | 426.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 431.24 | 426.17 | 426.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 431.30 | 427.20 | 427.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 435.45 | 428.85 | 427.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 442.33 | 443.34 | 440.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 442.33 | 443.34 | 440.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 445.88 | 443.49 | 441.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:15:00 | 446.15 | 443.49 | 441.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 438.09 | 442.34 | 441.13 | SL hit (close<static) qty=1.00 sl=440.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 435.91 | 440.08 | 440.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 434.18 | 438.05 | 439.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 435.52 | 435.40 | 436.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 10:45:00 | 435.91 | 435.40 | 436.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 436.12 | 435.60 | 436.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 436.76 | 435.60 | 436.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 436.00 | 435.68 | 436.62 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 444.94 | 437.43 | 437.17 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 435.91 | 438.79 | 438.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 434.76 | 437.98 | 438.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 12:15:00 | 436.48 | 436.08 | 437.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 436.48 | 436.08 | 437.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 436.48 | 436.08 | 437.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 436.48 | 436.08 | 437.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 435.91 | 436.05 | 436.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 436.64 | 436.05 | 436.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 440.00 | 436.52 | 436.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 440.00 | 436.52 | 436.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 438.00 | 436.82 | 437.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:30:00 | 437.82 | 436.90 | 437.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:00:00 | 437.24 | 436.90 | 437.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 12:15:00 | 439.45 | 437.41 | 437.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 439.45 | 437.41 | 437.25 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 434.00 | 437.29 | 437.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 431.42 | 434.83 | 435.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 429.03 | 428.64 | 430.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 429.03 | 428.64 | 430.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 429.03 | 428.64 | 430.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 430.91 | 428.64 | 430.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 429.45 | 428.80 | 430.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 431.09 | 428.80 | 430.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 429.73 | 429.07 | 430.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 429.73 | 429.07 | 430.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 430.45 | 429.35 | 430.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 430.00 | 429.35 | 430.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 428.97 | 429.27 | 430.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:45:00 | 427.45 | 428.99 | 429.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 427.73 | 429.95 | 430.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 430.73 | 430.15 | 430.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 430.73 | 430.15 | 430.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 436.94 | 431.73 | 430.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 15:15:00 | 445.21 | 445.34 | 441.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 09:15:00 | 443.12 | 445.34 | 441.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 441.52 | 444.58 | 441.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 441.52 | 444.58 | 441.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 440.39 | 443.74 | 441.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 440.39 | 443.74 | 441.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 438.85 | 442.76 | 441.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 438.85 | 442.76 | 441.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 431.39 | 439.74 | 440.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 429.61 | 435.89 | 438.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 433.18 | 431.82 | 434.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 433.18 | 431.82 | 434.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 407.30 | 407.12 | 409.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 405.82 | 409.13 | 409.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 413.33 | 408.60 | 408.80 | SL hit (close>static) qty=1.00 sl=410.91 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 415.00 | 409.88 | 409.36 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 408.76 | 409.54 | 409.57 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 11:15:00 | 409.88 | 409.60 | 409.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 410.70 | 409.82 | 409.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 409.09 | 409.68 | 409.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 409.09 | 409.68 | 409.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 409.09 | 409.68 | 409.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 409.09 | 409.68 | 409.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 14:15:00 | 409.09 | 409.56 | 409.59 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 410.39 | 409.60 | 409.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 412.48 | 410.44 | 409.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 416.88 | 417.01 | 415.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:00:00 | 416.88 | 417.01 | 415.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 414.30 | 416.55 | 415.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 414.30 | 416.55 | 415.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 414.70 | 416.18 | 415.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:15:00 | 415.45 | 416.18 | 415.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 414.24 | 415.04 | 415.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 414.24 | 415.04 | 415.13 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 418.67 | 415.69 | 415.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 421.33 | 417.91 | 416.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 418.42 | 419.22 | 417.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 15:00:00 | 418.42 | 419.22 | 417.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 418.12 | 418.86 | 417.92 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 415.64 | 417.24 | 417.45 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 418.61 | 417.63 | 417.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 420.45 | 418.55 | 418.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 10:15:00 | 418.85 | 418.90 | 418.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 11:00:00 | 418.85 | 418.90 | 418.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 418.94 | 418.91 | 418.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 418.94 | 418.91 | 418.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 418.00 | 419.22 | 418.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:30:00 | 420.33 | 418.75 | 418.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 13:15:00 | 419.55 | 418.75 | 418.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:00:00 | 421.03 | 419.20 | 418.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 417.15 | 418.95 | 418.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 417.15 | 418.95 | 418.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 415.15 | 418.19 | 418.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 411.58 | 410.74 | 413.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 411.58 | 410.74 | 413.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 413.76 | 411.34 | 413.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 413.76 | 411.34 | 413.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 414.48 | 411.97 | 413.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 414.67 | 411.97 | 413.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 414.21 | 412.42 | 413.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 412.00 | 413.16 | 413.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 415.06 | 413.01 | 412.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 415.06 | 413.01 | 412.97 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 412.45 | 413.14 | 413.18 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 415.73 | 413.51 | 413.28 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 410.70 | 412.95 | 413.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 409.97 | 412.36 | 412.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 417.67 | 412.17 | 412.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 417.67 | 412.17 | 412.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 417.67 | 412.17 | 412.53 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 418.58 | 413.45 | 413.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 425.55 | 417.97 | 415.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 418.18 | 422.02 | 419.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 418.18 | 422.02 | 419.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 418.18 | 422.02 | 419.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 418.18 | 422.02 | 419.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 418.91 | 421.40 | 419.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:45:00 | 419.97 | 420.94 | 419.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 416.09 | 419.97 | 419.11 | SL hit (close<static) qty=1.00 sl=417.27 alert=retest2 |

### Cycle 95 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 416.70 | 418.52 | 418.59 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 420.85 | 418.99 | 418.80 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 416.30 | 418.28 | 418.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 415.09 | 417.64 | 418.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 416.73 | 416.43 | 417.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 416.73 | 416.43 | 417.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 416.73 | 416.43 | 417.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 416.58 | 416.43 | 417.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 418.94 | 416.93 | 417.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 418.94 | 416.93 | 417.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 418.88 | 417.32 | 417.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 418.79 | 417.32 | 417.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 419.48 | 417.75 | 417.71 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 405.42 | 415.53 | 416.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 403.00 | 407.03 | 411.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 10:15:00 | 407.36 | 407.10 | 410.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 11:00:00 | 407.36 | 407.10 | 410.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 395.94 | 396.18 | 398.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:15:00 | 389.91 | 396.17 | 396.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 393.52 | 390.29 | 390.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 396.27 | 391.49 | 391.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 396.27 | 391.49 | 391.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 399.58 | 394.87 | 393.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 395.94 | 396.22 | 394.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 395.94 | 396.22 | 394.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 399.21 | 400.43 | 398.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 398.91 | 400.43 | 398.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 401.58 | 400.66 | 398.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 402.36 | 400.93 | 399.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:30:00 | 402.36 | 401.37 | 399.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 413.36 | 416.32 | 416.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 413.36 | 416.32 | 416.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 412.24 | 414.88 | 415.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 414.94 | 414.04 | 414.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 414.94 | 414.04 | 414.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 414.94 | 414.04 | 414.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 415.33 | 414.04 | 414.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 416.24 | 414.48 | 415.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 416.24 | 414.48 | 415.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 416.58 | 414.90 | 415.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 416.58 | 414.90 | 415.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 417.03 | 415.66 | 415.52 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 413.55 | 415.33 | 415.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 406.00 | 412.03 | 413.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 411.27 | 410.28 | 412.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 13:15:00 | 411.27 | 410.28 | 412.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 411.27 | 410.28 | 412.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 411.27 | 410.28 | 412.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 406.85 | 409.36 | 411.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 405.33 | 408.19 | 409.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 414.48 | 409.71 | 409.94 | SL hit (close>static) qty=1.00 sl=412.82 alert=retest2 |

### Cycle 104 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 417.45 | 411.26 | 410.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 418.39 | 412.68 | 411.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 415.39 | 416.55 | 414.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 415.39 | 416.55 | 414.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 414.94 | 416.23 | 414.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 416.64 | 415.93 | 414.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:00:00 | 417.00 | 416.14 | 414.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:45:00 | 416.52 | 416.22 | 414.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:30:00 | 418.09 | 416.71 | 415.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 416.70 | 419.38 | 418.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 416.70 | 419.38 | 418.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 416.97 | 418.90 | 418.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 420.79 | 418.90 | 418.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 414.76 | 418.31 | 417.99 | SL hit (close<static) qty=1.00 sl=415.58 alert=retest2 |

### Cycle 105 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 415.61 | 417.77 | 417.77 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 418.21 | 417.86 | 417.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 15:15:00 | 420.24 | 418.48 | 418.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 11:15:00 | 429.70 | 432.95 | 430.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 429.70 | 432.95 | 430.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 429.70 | 432.95 | 430.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 429.70 | 432.95 | 430.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 429.88 | 432.34 | 430.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:15:00 | 430.15 | 432.34 | 430.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 429.76 | 431.82 | 430.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 430.61 | 431.35 | 430.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 427.82 | 429.89 | 429.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 427.82 | 429.89 | 429.91 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 432.52 | 430.12 | 429.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 433.94 | 431.28 | 430.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 10:15:00 | 432.27 | 432.52 | 431.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 11:00:00 | 432.27 | 432.52 | 431.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 432.06 | 432.43 | 431.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 433.15 | 432.74 | 432.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 430.73 | 432.68 | 432.21 | SL hit (close<static) qty=1.00 sl=431.55 alert=retest2 |

### Cycle 109 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 431.12 | 433.39 | 433.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 15:15:00 | 428.48 | 430.35 | 431.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 424.88 | 424.70 | 427.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:30:00 | 424.48 | 424.70 | 427.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 419.73 | 423.90 | 425.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 415.42 | 419.64 | 423.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 413.67 | 409.55 | 409.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 413.67 | 409.55 | 409.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 420.21 | 411.68 | 410.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 432.76 | 433.16 | 427.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 432.76 | 433.16 | 427.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 432.76 | 433.16 | 427.81 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 424.55 | 427.69 | 427.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 422.79 | 426.71 | 427.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 412.67 | 411.26 | 416.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 412.67 | 411.26 | 416.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 415.73 | 413.22 | 415.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 415.55 | 413.22 | 415.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 410.91 | 412.76 | 415.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 403.91 | 412.02 | 414.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 403.55 | 407.28 | 411.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:15:00 | 383.71 | 402.33 | 407.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:15:00 | 383.37 | 402.33 | 407.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 415.65 | 404.99 | 408.07 | SL hit (close>ema200) qty=0.50 sl=404.99 alert=retest2 |

### Cycle 112 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 399.50 | 398.06 | 397.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 403.00 | 399.84 | 398.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 404.70 | 405.21 | 403.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 404.70 | 405.21 | 403.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 402.55 | 404.53 | 403.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 401.55 | 404.53 | 403.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 403.85 | 404.39 | 403.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 407.90 | 404.21 | 403.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 409.30 | 411.98 | 412.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 409.30 | 411.98 | 412.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 405.85 | 410.30 | 411.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 408.45 | 408.12 | 409.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 408.45 | 408.12 | 409.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 408.45 | 408.12 | 409.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 408.45 | 408.12 | 409.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 407.60 | 408.01 | 409.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 404.20 | 407.97 | 409.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 411.00 | 407.50 | 407.56 | SL hit (close>static) qty=1.00 sl=409.45 alert=retest2 |

### Cycle 114 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 411.80 | 408.36 | 407.94 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 403.05 | 407.26 | 407.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 402.05 | 404.23 | 405.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 362.75 | 361.54 | 365.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 362.75 | 361.54 | 365.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 360.30 | 356.16 | 358.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 360.20 | 356.16 | 358.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 357.75 | 356.48 | 358.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 355.90 | 356.41 | 357.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 360.85 | 358.22 | 358.30 | SL hit (close>static) qty=1.00 sl=360.45 alert=retest2 |

### Cycle 116 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 359.85 | 358.55 | 358.44 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 357.75 | 358.30 | 358.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 355.85 | 357.84 | 358.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 15:15:00 | 357.20 | 357.16 | 357.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:15:00 | 361.10 | 357.16 | 357.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 118 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 362.25 | 358.18 | 358.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 364.45 | 360.38 | 359.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 360.35 | 361.34 | 360.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 360.35 | 361.34 | 360.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 360.35 | 361.34 | 360.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 360.25 | 361.34 | 360.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 360.15 | 361.10 | 360.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:00:00 | 361.60 | 361.20 | 360.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 361.35 | 361.14 | 360.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 361.70 | 361.14 | 360.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 362.00 | 361.31 | 360.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 357.15 | 360.53 | 360.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 357.15 | 360.53 | 360.26 | SL hit (close<static) qty=1.00 sl=360.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 356.60 | 359.75 | 359.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 355.00 | 358.80 | 359.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 344.80 | 344.25 | 346.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 344.80 | 344.25 | 346.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 346.40 | 344.65 | 346.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 346.40 | 344.65 | 346.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 347.15 | 345.15 | 346.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 347.70 | 345.15 | 346.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 346.95 | 345.72 | 346.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 347.50 | 345.72 | 346.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 345.65 | 345.77 | 346.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 345.85 | 345.77 | 346.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 345.50 | 345.71 | 346.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 346.20 | 345.71 | 346.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 346.50 | 345.87 | 346.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 346.45 | 345.87 | 346.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 348.75 | 346.45 | 346.39 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 345.00 | 346.39 | 346.40 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 347.05 | 346.38 | 346.35 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 344.95 | 346.19 | 346.29 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 348.20 | 346.22 | 346.22 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 340.75 | 345.36 | 345.92 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 350.70 | 346.41 | 346.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 354.55 | 350.73 | 348.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 360.55 | 361.90 | 358.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 360.55 | 361.90 | 358.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 361.20 | 361.36 | 358.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 360.95 | 361.36 | 358.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 359.30 | 360.60 | 359.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 359.15 | 360.60 | 359.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 359.25 | 360.33 | 359.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 359.70 | 360.33 | 359.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 356.90 | 359.65 | 358.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 356.90 | 359.65 | 358.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 356.00 | 358.92 | 358.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 356.00 | 358.92 | 358.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 356.10 | 358.35 | 358.47 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 360.25 | 358.61 | 358.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 362.85 | 361.24 | 360.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 367.00 | 371.16 | 369.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 367.00 | 371.16 | 369.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 367.00 | 371.16 | 369.91 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 367.40 | 368.96 | 369.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 362.30 | 367.49 | 368.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 350.00 | 349.26 | 352.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 14:45:00 | 350.00 | 349.26 | 352.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 351.40 | 349.56 | 351.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 352.20 | 349.56 | 351.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 353.05 | 350.26 | 351.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 353.05 | 350.26 | 351.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 353.15 | 350.84 | 351.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 353.80 | 350.84 | 351.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 349.65 | 350.79 | 351.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 348.60 | 350.79 | 351.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 349.00 | 350.62 | 351.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 353.65 | 350.97 | 351.54 | SL hit (close>static) qty=1.00 sl=352.10 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 357.20 | 352.22 | 352.06 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 343.10 | 351.90 | 352.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 338.25 | 341.97 | 345.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 342.60 | 340.49 | 343.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:00:00 | 342.60 | 340.49 | 343.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 341.10 | 340.62 | 343.41 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 347.50 | 343.61 | 343.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 349.70 | 346.51 | 345.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 344.75 | 346.24 | 345.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 344.75 | 346.24 | 345.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 344.75 | 346.24 | 345.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 345.00 | 346.24 | 345.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 345.90 | 346.17 | 345.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:15:00 | 344.85 | 346.17 | 345.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 343.00 | 345.54 | 345.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 342.85 | 345.54 | 345.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 343.95 | 345.22 | 344.94 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 340.60 | 344.24 | 344.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 336.10 | 341.30 | 342.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 340.85 | 340.83 | 342.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:30:00 | 340.75 | 340.83 | 342.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 336.20 | 339.37 | 340.79 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 350.90 | 343.21 | 342.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 352.30 | 345.02 | 343.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 351.35 | 352.11 | 349.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:30:00 | 352.95 | 352.11 | 349.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 348.35 | 351.35 | 348.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 347.80 | 351.35 | 348.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 345.00 | 350.08 | 348.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 345.00 | 350.08 | 348.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 368.35 | 372.05 | 367.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 365.90 | 372.05 | 367.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 366.95 | 371.03 | 367.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 366.30 | 371.03 | 367.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 367.20 | 370.26 | 367.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 366.60 | 370.26 | 367.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 374.00 | 371.01 | 368.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 374.65 | 371.58 | 369.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 375.60 | 371.58 | 369.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 374.20 | 373.17 | 370.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:00:00 | 374.30 | 371.43 | 370.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 384.55 | 383.75 | 380.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 385.85 | 383.75 | 380.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 375.50 | 381.85 | 381.43 | SL hit (close<static) qty=1.00 sl=380.45 alert=retest2 |

### Cycle 135 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 379.00 | 381.21 | 381.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 377.00 | 379.96 | 380.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 378.15 | 378.00 | 379.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 378.15 | 378.00 | 379.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 380.85 | 378.57 | 379.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 380.85 | 378.57 | 379.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 380.55 | 378.97 | 379.44 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 382.95 | 380.24 | 379.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 383.20 | 380.84 | 380.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 11:15:00 | 379.40 | 380.92 | 380.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 11:15:00 | 379.40 | 380.92 | 380.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 379.40 | 380.92 | 380.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:00:00 | 379.40 | 380.92 | 380.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 380.90 | 380.92 | 380.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:45:00 | 381.60 | 381.09 | 380.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 381.30 | 382.15 | 381.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 13:15:00 | 378.90 | 381.30 | 381.17 | SL hit (close<static) qty=1.00 sl=379.10 alert=retest2 |

### Cycle 137 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 375.85 | 380.21 | 380.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 375.20 | 379.21 | 380.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 378.25 | 378.11 | 379.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 14:15:00 | 378.25 | 378.11 | 379.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 378.25 | 378.11 | 379.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 379.65 | 378.11 | 379.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 379.50 | 378.42 | 379.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 377.50 | 378.76 | 378.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 374.05 | 378.82 | 378.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 376.45 | 378.18 | 378.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 377.25 | 378.57 | 378.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 382.50 | 379.36 | 379.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 382.50 | 379.36 | 379.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 383.95 | 380.28 | 379.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 389.05 | 389.06 | 385.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:45:00 | 389.05 | 389.06 | 385.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 381.85 | 387.00 | 385.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 381.85 | 387.00 | 385.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 383.50 | 386.30 | 385.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 374.60 | 386.30 | 385.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 373.40 | 383.72 | 384.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 351.75 | 364.69 | 372.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 354.00 | 352.56 | 359.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 354.00 | 352.56 | 359.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 343.55 | 339.32 | 342.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 343.55 | 339.32 | 342.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 344.70 | 340.40 | 342.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 345.05 | 340.40 | 342.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 315.50 | 314.43 | 317.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 313.60 | 314.43 | 317.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 313.85 | 314.49 | 317.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 12:15:00 | 317.95 | 315.18 | 317.50 | SL hit (close>static) qty=1.00 sl=317.75 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 325.95 | 319.03 | 318.75 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 313.25 | 320.08 | 320.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 313.10 | 318.68 | 319.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 314.90 | 313.67 | 316.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:45:00 | 314.65 | 313.67 | 316.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 314.15 | 314.03 | 315.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 314.15 | 314.03 | 315.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 307.30 | 307.25 | 310.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:45:00 | 305.85 | 307.11 | 309.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 319.20 | 311.69 | 311.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 319.20 | 311.69 | 311.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 321.80 | 313.71 | 312.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 304.15 | 314.51 | 313.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 304.15 | 314.51 | 313.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 304.15 | 314.51 | 313.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 301.80 | 314.51 | 313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 303.75 | 312.35 | 312.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 302.05 | 310.29 | 311.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 303.65 | 300.78 | 304.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 303.65 | 300.78 | 304.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 303.65 | 300.78 | 304.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 302.70 | 300.78 | 304.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 302.60 | 301.15 | 303.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 302.60 | 302.74 | 304.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 303.00 | 302.70 | 303.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 302.95 | 300.08 | 301.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 302.95 | 300.08 | 301.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 303.40 | 300.74 | 301.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 303.40 | 300.74 | 301.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 303.40 | 301.27 | 302.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 302.00 | 301.27 | 302.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 304.05 | 302.42 | 302.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 305.55 | 302.42 | 302.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 304.90 | 302.91 | 302.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 304.90 | 302.91 | 302.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 307.10 | 303.75 | 303.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 305.30 | 305.39 | 304.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 305.30 | 305.39 | 304.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 305.30 | 305.39 | 304.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 307.35 | 305.37 | 304.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 307.20 | 305.37 | 304.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 338.09 | 330.09 | 320.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 353.10 | 357.53 | 357.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 350.80 | 354.13 | 356.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 351.30 | 350.88 | 353.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 351.30 | 350.88 | 353.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 351.30 | 350.88 | 353.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:45:00 | 350.20 | 352.04 | 352.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 360.05 | 353.41 | 353.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 360.05 | 353.41 | 353.17 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 343.40 | 352.24 | 353.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 339.40 | 348.32 | 351.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 343.95 | 343.68 | 346.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 343.95 | 343.68 | 346.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 346.35 | 342.50 | 343.68 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 351.45 | 345.46 | 344.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 358.00 | 347.97 | 345.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 355.05 | 356.88 | 353.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 355.55 | 356.88 | 353.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 09:15:00 | 584.24 | 2024-05-27 09:15:00 | 580.15 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-05-27 10:45:00 | 583.03 | 2024-05-28 09:15:00 | 578.73 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-05-30 11:00:00 | 568.36 | 2024-06-03 12:15:00 | 574.67 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-06-03 10:30:00 | 566.94 | 2024-06-03 12:15:00 | 574.67 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-06-13 12:15:00 | 599.48 | 2024-06-19 09:15:00 | 595.76 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-06-14 10:15:00 | 600.06 | 2024-06-19 09:15:00 | 595.76 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-06-14 11:00:00 | 599.82 | 2024-06-19 09:15:00 | 595.76 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-06-18 12:00:00 | 599.42 | 2024-06-19 09:15:00 | 595.76 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-06-20 12:45:00 | 592.27 | 2024-06-27 14:15:00 | 590.39 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-06-21 09:15:00 | 587.97 | 2024-06-27 14:15:00 | 590.39 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-07-08 09:15:00 | 613.21 | 2024-07-18 10:15:00 | 612.27 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-07-23 12:15:00 | 591.45 | 2024-07-24 09:15:00 | 623.48 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2024-07-23 13:30:00 | 603.24 | 2024-07-24 09:15:00 | 623.48 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-08-29 11:45:00 | 649.24 | 2024-08-29 14:15:00 | 680.03 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2024-09-05 11:15:00 | 650.94 | 2024-09-11 09:15:00 | 618.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 14:00:00 | 650.70 | 2024-09-11 09:15:00 | 618.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 11:15:00 | 650.94 | 2024-09-12 09:15:00 | 585.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-05 14:00:00 | 650.70 | 2024-09-12 09:15:00 | 585.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-07 10:45:00 | 560.70 | 2024-10-09 11:15:00 | 570.97 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-10-07 12:45:00 | 561.00 | 2024-10-09 11:15:00 | 570.97 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-10-07 14:45:00 | 562.52 | 2024-10-09 11:15:00 | 570.97 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-10-08 09:15:00 | 544.85 | 2024-10-09 11:15:00 | 570.97 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2024-10-21 11:30:00 | 547.94 | 2024-10-25 10:15:00 | 520.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 547.82 | 2024-10-25 10:15:00 | 520.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 547.94 | 2024-10-28 09:15:00 | 526.88 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2024-10-21 14:45:00 | 547.82 | 2024-10-28 09:15:00 | 526.88 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2024-11-04 09:15:00 | 502.33 | 2024-11-06 12:15:00 | 508.33 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-11 13:15:00 | 494.58 | 2024-11-14 12:15:00 | 469.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:15:00 | 494.58 | 2024-11-18 11:15:00 | 469.88 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2024-12-05 13:15:00 | 476.42 | 2024-12-05 13:15:00 | 481.36 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-10 09:15:00 | 488.48 | 2024-12-12 09:15:00 | 479.52 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-12-10 12:45:00 | 485.70 | 2024-12-12 09:15:00 | 479.52 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-12-10 15:15:00 | 485.45 | 2024-12-12 09:15:00 | 479.52 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-12-16 10:15:00 | 476.67 | 2024-12-19 09:15:00 | 452.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 476.45 | 2024-12-19 09:15:00 | 452.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 13:30:00 | 476.00 | 2024-12-19 09:15:00 | 452.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 476.06 | 2024-12-19 09:15:00 | 452.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 13:15:00 | 474.91 | 2024-12-19 09:15:00 | 451.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 14:15:00 | 474.24 | 2024-12-19 09:15:00 | 450.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 476.67 | 2024-12-24 09:15:00 | 446.06 | STOP_HIT | 0.50 | 6.42% |
| SELL | retest2 | 2024-12-16 10:45:00 | 476.45 | 2024-12-24 09:15:00 | 446.06 | STOP_HIT | 0.50 | 6.38% |
| SELL | retest2 | 2024-12-16 13:30:00 | 476.00 | 2024-12-24 09:15:00 | 446.06 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2024-12-17 10:15:00 | 476.06 | 2024-12-24 09:15:00 | 446.06 | STOP_HIT | 0.50 | 6.30% |
| SELL | retest2 | 2024-12-17 13:15:00 | 474.91 | 2024-12-24 09:15:00 | 446.06 | STOP_HIT | 0.50 | 6.07% |
| SELL | retest2 | 2024-12-17 14:15:00 | 474.24 | 2024-12-24 09:15:00 | 446.06 | STOP_HIT | 0.50 | 5.94% |
| BUY | retest2 | 2025-01-07 09:15:00 | 471.52 | 2025-01-09 14:15:00 | 472.30 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-01-07 10:00:00 | 473.03 | 2025-01-09 14:15:00 | 472.30 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-01-09 11:00:00 | 472.36 | 2025-01-09 14:15:00 | 472.30 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-01-10 14:00:00 | 470.48 | 2025-01-16 11:15:00 | 471.94 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-01-20 11:30:00 | 471.91 | 2025-01-21 10:15:00 | 467.61 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-01-20 13:00:00 | 471.73 | 2025-01-21 10:15:00 | 467.61 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-01-21 09:15:00 | 474.09 | 2025-01-21 10:15:00 | 467.61 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-23 14:15:00 | 457.30 | 2025-01-27 10:15:00 | 434.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 457.30 | 2025-01-28 10:15:00 | 436.58 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2025-02-03 09:15:00 | 420.52 | 2025-02-04 14:15:00 | 430.36 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-02-14 10:45:00 | 410.88 | 2025-02-17 15:15:00 | 416.33 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-02-14 15:15:00 | 411.24 | 2025-02-17 15:15:00 | 416.33 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-03-12 09:15:00 | 402.30 | 2025-03-27 09:15:00 | 406.42 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-04-08 10:30:00 | 355.39 | 2025-04-15 09:15:00 | 378.18 | STOP_HIT | 1.00 | -6.41% |
| SELL | retest2 | 2025-04-08 11:15:00 | 355.21 | 2025-04-15 09:15:00 | 378.18 | STOP_HIT | 1.00 | -6.47% |
| SELL | retest2 | 2025-04-09 09:15:00 | 356.36 | 2025-04-15 09:15:00 | 378.18 | STOP_HIT | 1.00 | -6.12% |
| SELL | retest2 | 2025-04-09 10:00:00 | 356.06 | 2025-04-15 09:15:00 | 378.18 | STOP_HIT | 1.00 | -6.21% |
| SELL | retest2 | 2025-04-11 10:15:00 | 363.45 | 2025-04-15 09:15:00 | 378.18 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-04-11 12:30:00 | 362.97 | 2025-04-15 09:15:00 | 378.18 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2025-04-25 12:30:00 | 398.45 | 2025-04-30 09:15:00 | 390.91 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-04-28 09:15:00 | 398.48 | 2025-04-30 09:15:00 | 390.91 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-05-02 11:15:00 | 395.52 | 2025-05-05 10:15:00 | 401.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-09 14:00:00 | 428.79 | 2025-05-14 11:15:00 | 422.27 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-05-09 15:00:00 | 429.39 | 2025-05-14 11:15:00 | 422.27 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-14 10:15:00 | 428.12 | 2025-05-14 11:15:00 | 422.27 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-05-20 10:15:00 | 446.15 | 2025-05-20 12:15:00 | 438.09 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-05-29 11:30:00 | 437.82 | 2025-05-29 12:15:00 | 439.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-05-29 12:00:00 | 437.24 | 2025-05-29 12:15:00 | 439.45 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-06-05 10:45:00 | 427.45 | 2025-06-06 13:15:00 | 430.73 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-06 09:15:00 | 427.73 | 2025-06-06 13:15:00 | 430.73 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-23 09:15:00 | 405.82 | 2025-06-24 09:15:00 | 413.33 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-07-01 12:15:00 | 415.45 | 2025-07-01 15:15:00 | 414.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-07-10 12:30:00 | 420.33 | 2025-07-11 11:15:00 | 417.15 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-07-10 13:15:00 | 419.55 | 2025-07-11 11:15:00 | 417.15 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-10 14:00:00 | 421.03 | 2025-07-11 11:15:00 | 417.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-07-16 09:15:00 | 412.00 | 2025-07-17 11:15:00 | 415.06 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-25 11:45:00 | 419.97 | 2025-07-25 12:15:00 | 416.09 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-08-07 09:15:00 | 389.91 | 2025-08-11 12:15:00 | 396.27 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-08-11 11:30:00 | 393.52 | 2025-08-11 12:15:00 | 396.27 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-14 12:45:00 | 402.36 | 2025-08-22 10:15:00 | 413.36 | STOP_HIT | 1.00 | 2.73% |
| BUY | retest2 | 2025-08-14 14:30:00 | 402.36 | 2025-08-22 10:15:00 | 413.36 | STOP_HIT | 1.00 | 2.73% |
| SELL | retest2 | 2025-08-29 15:00:00 | 405.33 | 2025-09-01 12:15:00 | 414.48 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-09-03 09:15:00 | 416.64 | 2025-09-05 11:15:00 | 414.76 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-09-03 10:00:00 | 417.00 | 2025-09-05 12:15:00 | 415.61 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-09-03 10:45:00 | 416.52 | 2025-09-05 12:15:00 | 415.61 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-09-03 11:30:00 | 418.09 | 2025-09-05 12:15:00 | 415.61 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-05 09:15:00 | 420.79 | 2025-09-05 12:15:00 | 415.61 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-09-10 15:15:00 | 430.61 | 2025-09-11 11:15:00 | 427.82 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-09-16 09:30:00 | 433.15 | 2025-09-16 12:15:00 | 430.73 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-16 15:15:00 | 433.27 | 2025-09-18 13:15:00 | 431.21 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-24 14:15:00 | 415.42 | 2025-09-30 15:15:00 | 413.67 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-10-13 09:15:00 | 403.91 | 2025-10-14 09:15:00 | 383.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 12:45:00 | 403.55 | 2025-10-14 09:15:00 | 383.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 403.91 | 2025-10-14 10:15:00 | 415.65 | STOP_HIT | 0.50 | -2.91% |
| SELL | retest2 | 2025-10-13 12:45:00 | 403.55 | 2025-10-14 10:15:00 | 415.65 | STOP_HIT | 0.50 | -3.00% |
| SELL | retest2 | 2025-10-14 13:45:00 | 400.80 | 2025-10-20 11:15:00 | 399.50 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-10-27 09:15:00 | 407.90 | 2025-11-04 11:15:00 | 409.30 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-11-07 09:15:00 | 404.20 | 2025-11-10 11:15:00 | 411.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-11-26 11:30:00 | 355.90 | 2025-11-27 10:15:00 | 360.85 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-12-02 12:00:00 | 361.60 | 2025-12-03 09:15:00 | 357.15 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-02 13:45:00 | 361.35 | 2025-12-03 09:15:00 | 357.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-02 14:15:00 | 361.70 | 2025-12-03 09:15:00 | 357.15 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-02 15:00:00 | 362.00 | 2025-12-03 09:15:00 | 357.15 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-14 14:15:00 | 348.60 | 2026-01-16 09:15:00 | 353.65 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-14 15:15:00 | 349.00 | 2026-01-16 09:15:00 | 353.65 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-05 14:45:00 | 374.65 | 2026-02-13 09:15:00 | 375.50 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2026-02-05 15:15:00 | 375.60 | 2026-02-13 15:15:00 | 379.00 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2026-02-06 11:00:00 | 374.20 | 2026-02-13 15:15:00 | 379.00 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2026-02-09 10:00:00 | 374.30 | 2026-02-13 15:15:00 | 379.00 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2026-02-12 10:15:00 | 385.85 | 2026-02-13 15:15:00 | 379.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-02-13 12:15:00 | 385.75 | 2026-02-13 15:15:00 | 379.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-18 13:45:00 | 381.60 | 2026-02-19 13:15:00 | 378.90 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-02-19 12:15:00 | 381.30 | 2026-02-19 13:15:00 | 378.90 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-02-24 10:45:00 | 377.50 | 2026-02-25 10:15:00 | 382.50 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-24 12:15:00 | 374.05 | 2026-02-25 10:15:00 | 382.50 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-02-24 14:30:00 | 376.45 | 2026-02-25 10:15:00 | 382.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-25 09:30:00 | 377.25 | 2026-02-25 10:15:00 | 382.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-17 11:15:00 | 313.60 | 2026-03-17 12:15:00 | 317.95 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-17 12:15:00 | 313.85 | 2026-03-17 12:15:00 | 317.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-03-24 10:45:00 | 305.85 | 2026-03-25 09:15:00 | 319.20 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-04-01 10:15:00 | 302.70 | 2026-04-06 11:15:00 | 304.90 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-04-01 11:00:00 | 302.60 | 2026-04-06 11:15:00 | 304.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-04-01 13:30:00 | 302.60 | 2026-04-06 11:15:00 | 304.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-04-01 14:30:00 | 303.00 | 2026-04-06 11:15:00 | 304.90 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-04-07 13:30:00 | 307.35 | 2026-04-09 09:15:00 | 338.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 14:15:00 | 307.20 | 2026-04-09 09:15:00 | 337.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 13:45:00 | 350.20 | 2026-04-29 09:15:00 | 360.05 | STOP_HIT | 1.00 | -2.81% |
