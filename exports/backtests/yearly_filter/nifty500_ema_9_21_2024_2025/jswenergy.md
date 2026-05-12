# JSW Energy Ltd. (JSWENERGY)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 573.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 100 |
| ALERT2 | 100 |
| ALERT2_SKIP | 52 |
| ALERT3 | 278 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 136 |
| PARTIAL | 12 |
| TARGET_HIT | 12 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 100
- **Target hits / Stop hits / Partials:** 12 / 127 / 12
- **Avg / median % per leg:** 0.32% / -1.03%
- **Sum % (uncompounded):** 49.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 16 | 23.2% | 7 | 62 | 0 | -0.16% | -10.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.05% | -1.0% |
| BUY @ 3rd Alert (retest2) | 68 | 16 | 23.5% | 7 | 61 | 0 | -0.14% | -9.7% |
| SELL (all) | 82 | 35 | 42.7% | 5 | 65 | 12 | 0.73% | 59.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.91% | -3.8% |
| SELL @ 3rd Alert (retest2) | 80 | 35 | 43.8% | 5 | 63 | 12 | 0.80% | 63.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.63% | -4.9% |
| retest2 (combined) | 148 | 51 | 34.5% | 12 | 124 | 12 | 0.36% | 53.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 570.80 | 563.47 | 562.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 572.65 | 565.31 | 563.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 598.45 | 598.86 | 591.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 09:45:00 | 599.75 | 598.86 | 591.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 593.00 | 597.17 | 592.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 593.00 | 597.17 | 592.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 591.55 | 596.05 | 592.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 591.55 | 596.05 | 592.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 593.00 | 595.44 | 592.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 607.70 | 595.44 | 592.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:00:00 | 605.80 | 597.51 | 593.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:15:00 | 600.20 | 599.51 | 595.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 590.00 | 596.89 | 595.57 | SL hit (close<static) qty=1.00 sl=591.20 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 597.30 | 610.97 | 612.43 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 620.90 | 612.01 | 611.87 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 608.95 | 612.52 | 612.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 12:15:00 | 606.85 | 610.77 | 611.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 10:15:00 | 607.10 | 606.78 | 609.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 10:15:00 | 607.10 | 606.78 | 609.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 607.10 | 606.78 | 609.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:45:00 | 611.80 | 606.78 | 609.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 609.35 | 607.29 | 609.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:45:00 | 608.80 | 607.29 | 609.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 620.00 | 609.84 | 610.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 620.00 | 609.84 | 610.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 13:15:00 | 613.65 | 610.60 | 610.54 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 608.00 | 610.08 | 610.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 601.70 | 608.20 | 609.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 615.85 | 609.73 | 609.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 615.85 | 609.73 | 609.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 615.85 | 609.73 | 609.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 615.85 | 609.73 | 609.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 619.50 | 611.68 | 610.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 629.70 | 616.69 | 613.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 614.00 | 616.15 | 613.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 614.00 | 616.15 | 613.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 614.00 | 616.15 | 613.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 614.00 | 616.15 | 613.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 617.90 | 616.50 | 613.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 643.00 | 616.50 | 613.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 584.30 | 632.83 | 629.15 | SL hit (close<static) qty=1.00 sl=609.65 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 545.75 | 615.41 | 621.57 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 625.80 | 603.42 | 601.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 634.35 | 624.68 | 617.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 688.00 | 697.47 | 684.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 688.00 | 697.47 | 684.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 688.00 | 697.47 | 684.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 695.40 | 697.47 | 684.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 690.10 | 695.99 | 684.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 688.45 | 695.99 | 684.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 717.70 | 728.39 | 724.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 717.70 | 728.39 | 724.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 721.50 | 727.01 | 724.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 718.00 | 727.01 | 724.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 721.45 | 723.40 | 722.82 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 718.00 | 722.32 | 722.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 712.20 | 719.14 | 720.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 729.65 | 720.09 | 720.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 729.65 | 720.09 | 720.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 729.65 | 720.09 | 720.90 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 734.65 | 723.00 | 722.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 15:15:00 | 738.00 | 729.71 | 726.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 733.70 | 734.35 | 730.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 733.70 | 734.35 | 730.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 731.50 | 733.78 | 730.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 739.50 | 733.78 | 730.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 729.90 | 741.51 | 738.33 | SL hit (close<static) qty=1.00 sl=730.30 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 728.90 | 736.58 | 736.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 10:15:00 | 724.45 | 731.96 | 734.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 14:15:00 | 734.90 | 730.19 | 732.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 14:15:00 | 734.90 | 730.19 | 732.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 734.90 | 730.19 | 732.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 734.90 | 730.19 | 732.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 731.30 | 730.41 | 732.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 733.05 | 730.41 | 732.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 733.35 | 731.00 | 732.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 11:30:00 | 726.30 | 730.25 | 731.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 14:15:00 | 726.55 | 729.65 | 731.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 14:15:00 | 737.30 | 730.33 | 729.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 737.30 | 730.33 | 729.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 739.80 | 733.09 | 731.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 737.90 | 738.24 | 735.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 737.90 | 738.24 | 735.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 737.90 | 738.24 | 735.31 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 13:15:00 | 729.20 | 733.77 | 733.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 15:15:00 | 720.00 | 730.05 | 732.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 726.25 | 718.39 | 723.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 726.25 | 718.39 | 723.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 726.25 | 718.39 | 723.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 726.25 | 718.39 | 723.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 723.50 | 719.41 | 723.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 733.80 | 719.41 | 723.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 734.95 | 722.52 | 724.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 738.25 | 722.52 | 724.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 724.85 | 722.98 | 724.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 735.50 | 722.98 | 724.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 727.15 | 723.82 | 725.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 727.15 | 723.82 | 725.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 722.85 | 723.62 | 724.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 726.20 | 723.62 | 724.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 714.60 | 721.82 | 723.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 713.00 | 719.12 | 722.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:45:00 | 711.45 | 717.17 | 721.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:30:00 | 711.45 | 717.38 | 720.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:30:00 | 712.65 | 716.12 | 719.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 707.00 | 712.61 | 716.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:45:00 | 699.15 | 711.31 | 713.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 11:00:00 | 699.25 | 708.90 | 712.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 11:30:00 | 698.80 | 707.52 | 711.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 12:30:00 | 698.20 | 705.32 | 708.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 702.00 | 703.88 | 706.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 715.30 | 703.88 | 706.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 705.85 | 704.27 | 706.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 711.40 | 707.97 | 707.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 711.40 | 707.97 | 707.76 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 699.10 | 706.10 | 706.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 672.95 | 688.61 | 695.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 685.60 | 678.73 | 686.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 685.60 | 678.73 | 686.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 685.60 | 678.73 | 686.18 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 694.55 | 690.36 | 689.98 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 684.40 | 689.24 | 689.69 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 15:15:00 | 691.50 | 689.83 | 689.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 722.85 | 696.43 | 692.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 13:15:00 | 731.95 | 732.32 | 723.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 14:00:00 | 731.95 | 732.32 | 723.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 727.45 | 731.35 | 723.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 728.50 | 731.35 | 723.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 731.80 | 731.44 | 724.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 719.70 | 731.44 | 724.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 717.70 | 728.69 | 723.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 717.70 | 728.69 | 723.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 719.60 | 726.87 | 723.46 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 696.20 | 717.43 | 719.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 15:15:00 | 681.00 | 689.59 | 697.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 686.25 | 685.29 | 691.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 13:45:00 | 684.25 | 685.29 | 691.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 689.80 | 686.99 | 691.43 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 712.80 | 694.71 | 694.03 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 687.70 | 698.78 | 699.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 681.05 | 689.98 | 694.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 657.15 | 656.57 | 668.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 664.65 | 660.26 | 665.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 664.65 | 660.26 | 665.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 664.65 | 660.26 | 665.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 663.40 | 660.89 | 665.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 666.45 | 660.89 | 665.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 672.70 | 663.25 | 666.19 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 669.45 | 667.90 | 667.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 685.00 | 672.19 | 669.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 710.45 | 712.00 | 700.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:45:00 | 712.60 | 712.00 | 700.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 709.95 | 708.74 | 702.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:30:00 | 707.35 | 708.74 | 702.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 705.00 | 708.33 | 705.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 705.00 | 708.33 | 705.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 703.50 | 707.37 | 705.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 710.20 | 707.37 | 705.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 705.10 | 706.39 | 705.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 13:15:00 | 718.60 | 724.89 | 725.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 718.60 | 724.89 | 725.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 712.05 | 721.29 | 722.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 721.20 | 720.11 | 721.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 721.20 | 720.11 | 721.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 721.20 | 720.11 | 721.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:30:00 | 712.05 | 717.30 | 720.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:15:00 | 676.45 | 696.12 | 704.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-04 13:15:00 | 692.90 | 692.12 | 699.58 | SL hit (close>ema200) qty=0.50 sl=692.12 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 13:15:00 | 720.50 | 703.91 | 702.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 09:15:00 | 727.20 | 713.26 | 708.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 743.10 | 747.77 | 738.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 743.10 | 747.77 | 738.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 741.90 | 746.59 | 739.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 737.55 | 746.59 | 739.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 759.70 | 764.48 | 758.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:45:00 | 759.70 | 764.48 | 758.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 763.50 | 764.28 | 759.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:30:00 | 756.85 | 764.28 | 759.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 771.55 | 768.39 | 763.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 765.40 | 768.39 | 763.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 770.00 | 768.71 | 764.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:30:00 | 771.50 | 768.71 | 764.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 746.30 | 764.23 | 762.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:00:00 | 746.30 | 764.23 | 762.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 13:15:00 | 750.15 | 761.41 | 761.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 13:15:00 | 736.10 | 746.56 | 751.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 757.55 | 748.23 | 750.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 757.55 | 748.23 | 750.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 757.55 | 748.23 | 750.65 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 765.60 | 753.82 | 752.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 768.60 | 756.78 | 754.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 786.60 | 791.42 | 782.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 786.60 | 791.42 | 782.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 785.90 | 790.09 | 783.77 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 778.25 | 782.74 | 782.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 776.00 | 781.39 | 782.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 728.55 | 728.38 | 740.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 14:00:00 | 728.55 | 728.38 | 740.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 701.70 | 684.50 | 693.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 701.70 | 684.50 | 693.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 697.85 | 687.17 | 694.19 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 723.50 | 700.51 | 699.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 725.50 | 711.22 | 704.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 726.80 | 728.35 | 720.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 726.80 | 728.35 | 720.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 714.55 | 725.77 | 721.54 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 712.10 | 719.04 | 719.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 706.85 | 716.60 | 718.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 696.95 | 696.24 | 701.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 10:00:00 | 696.95 | 696.24 | 701.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 688.85 | 691.97 | 696.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 12:00:00 | 685.50 | 690.46 | 695.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:45:00 | 683.90 | 679.42 | 683.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:15:00 | 682.50 | 679.42 | 683.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:00:00 | 685.35 | 680.61 | 683.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 683.20 | 681.13 | 683.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:30:00 | 685.35 | 681.13 | 683.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 679.00 | 680.70 | 683.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 12:15:00 | 673.60 | 680.67 | 682.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 651.23 | 674.22 | 677.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 649.70 | 674.22 | 677.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 651.08 | 674.22 | 677.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 11:45:00 | 662.50 | 670.26 | 675.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 13:15:00 | 669.00 | 668.85 | 673.76 | SL hit (close>ema200) qty=0.50 sl=668.85 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 682.45 | 677.31 | 676.92 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 14:15:00 | 670.55 | 676.25 | 676.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 646.80 | 669.36 | 673.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 669.45 | 663.19 | 668.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 14:15:00 | 669.45 | 663.19 | 668.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 669.45 | 663.19 | 668.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 669.45 | 663.19 | 668.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 667.20 | 663.99 | 668.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 665.00 | 663.99 | 668.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:00:00 | 665.30 | 663.78 | 667.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:30:00 | 664.90 | 663.23 | 666.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:45:00 | 663.85 | 654.65 | 659.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 668.45 | 657.41 | 660.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:00:00 | 668.45 | 657.41 | 660.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 677.70 | 661.47 | 661.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 13:15:00 | 677.70 | 661.47 | 661.83 | SL hit (close>static) qty=1.00 sl=674.30 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 680.35 | 665.24 | 663.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 15:15:00 | 688.00 | 669.79 | 665.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 671.30 | 673.48 | 669.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 671.30 | 673.48 | 669.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 674.50 | 678.84 | 675.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 674.50 | 678.84 | 675.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 675.35 | 678.14 | 675.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 678.90 | 678.14 | 675.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 14:15:00 | 662.50 | 673.51 | 674.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 662.50 | 673.51 | 674.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 15:15:00 | 662.20 | 671.25 | 673.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 678.05 | 662.89 | 666.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 678.05 | 662.89 | 666.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 678.05 | 662.89 | 666.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 678.05 | 662.89 | 666.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 680.55 | 666.42 | 667.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 680.55 | 666.42 | 667.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 679.40 | 669.02 | 668.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 15:15:00 | 684.95 | 676.37 | 672.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 15:15:00 | 745.95 | 748.24 | 736.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-13 09:15:00 | 768.00 | 748.24 | 736.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 739.30 | 749.04 | 742.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 14:00:00 | 739.30 | 749.04 | 742.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 727.55 | 744.74 | 740.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 15:00:00 | 727.55 | 744.74 | 740.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 734.90 | 742.77 | 740.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:15:00 | 740.30 | 742.77 | 740.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 10:15:00 | 731.10 | 738.61 | 738.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 727.40 | 732.79 | 735.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 14:15:00 | 701.45 | 696.81 | 706.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-21 15:00:00 | 701.45 | 696.81 | 706.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 698.00 | 695.34 | 700.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 15:00:00 | 674.30 | 692.18 | 697.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:15:00 | 640.58 | 644.71 | 648.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-04 12:15:00 | 647.10 | 645.19 | 648.50 | SL hit (close>ema200) qty=0.50 sl=645.19 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 663.60 | 650.65 | 649.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 15:15:00 | 665.95 | 659.10 | 654.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 11:15:00 | 673.40 | 673.71 | 667.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 12:00:00 | 673.40 | 673.71 | 667.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 674.15 | 673.46 | 668.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 15:15:00 | 676.10 | 673.46 | 668.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 10:15:00 | 665.80 | 671.52 | 669.06 | SL hit (close<static) qty=1.00 sl=667.50 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 681.25 | 683.43 | 683.55 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 684.90 | 682.64 | 682.54 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 674.00 | 681.45 | 682.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 668.10 | 677.82 | 680.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 655.15 | 636.60 | 641.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 655.15 | 636.60 | 641.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 655.15 | 636.60 | 641.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:45:00 | 651.00 | 639.27 | 642.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 649.70 | 641.41 | 641.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 649.70 | 641.41 | 641.12 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 637.35 | 641.77 | 642.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 635.25 | 638.66 | 640.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 543.35 | 531.36 | 543.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 543.35 | 531.36 | 543.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 543.35 | 531.36 | 543.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:45:00 | 542.25 | 531.36 | 543.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 552.30 | 535.55 | 544.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 552.30 | 535.55 | 544.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 547.35 | 537.91 | 544.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:45:00 | 544.70 | 539.73 | 544.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 559.20 | 547.39 | 547.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 559.20 | 547.39 | 547.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 561.80 | 550.27 | 548.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 14:15:00 | 568.20 | 570.29 | 565.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 568.20 | 570.29 | 565.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 566.00 | 569.43 | 565.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 563.55 | 569.43 | 565.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 564.55 | 568.46 | 565.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:45:00 | 570.45 | 568.78 | 566.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 12:15:00 | 561.00 | 567.05 | 567.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 561.00 | 567.05 | 567.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 557.40 | 564.41 | 566.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 551.35 | 551.08 | 556.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 551.35 | 551.08 | 556.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 552.20 | 551.31 | 555.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 556.40 | 551.31 | 555.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 555.25 | 552.09 | 555.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:15:00 | 550.70 | 552.45 | 555.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:00:00 | 550.10 | 551.98 | 555.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 550.15 | 551.35 | 554.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 550.25 | 551.92 | 553.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 523.16 | 533.45 | 541.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 522.60 | 533.45 | 541.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 522.64 | 533.45 | 541.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 522.74 | 533.45 | 541.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-29 09:15:00 | 495.63 | 504.51 | 521.15 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 13:15:00 | 503.40 | 496.64 | 495.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 508.55 | 499.02 | 497.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 478.70 | 500.02 | 499.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 478.70 | 500.02 | 499.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 478.70 | 500.02 | 499.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 478.70 | 500.02 | 499.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 482.35 | 496.48 | 497.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 14:15:00 | 472.05 | 491.60 | 495.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 456.05 | 455.02 | 469.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:45:00 | 458.65 | 455.02 | 469.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 467.50 | 459.90 | 467.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 467.75 | 459.90 | 467.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 469.00 | 461.72 | 467.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 469.00 | 461.72 | 467.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 471.00 | 463.57 | 467.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 488.70 | 463.57 | 467.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 495.60 | 474.09 | 471.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 500.25 | 482.57 | 476.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 15:15:00 | 492.80 | 492.84 | 487.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 09:15:00 | 488.30 | 492.84 | 487.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 488.95 | 492.06 | 487.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 488.95 | 492.06 | 487.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 490.30 | 491.71 | 487.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 492.30 | 491.71 | 487.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 477.35 | 485.47 | 486.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 477.35 | 485.47 | 486.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 474.35 | 479.81 | 482.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 469.80 | 466.03 | 471.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 469.80 | 466.03 | 471.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 464.75 | 465.78 | 470.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 459.05 | 470.10 | 470.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 436.10 | 452.93 | 461.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 432.50 | 431.72 | 443.44 | SL hit (close>ema200) qty=0.50 sl=431.72 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 455.70 | 441.47 | 441.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 459.00 | 444.98 | 442.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 488.00 | 489.98 | 478.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 14:15:00 | 479.90 | 486.17 | 480.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 479.90 | 486.17 | 480.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 479.90 | 486.17 | 480.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 480.05 | 484.95 | 480.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 474.30 | 484.95 | 480.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 477.00 | 483.36 | 480.34 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 12:15:00 | 469.95 | 477.06 | 477.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 467.50 | 473.68 | 476.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 463.30 | 454.60 | 460.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 463.30 | 454.60 | 460.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 463.30 | 454.60 | 460.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 463.30 | 454.60 | 460.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 462.70 | 456.22 | 460.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 462.10 | 456.22 | 460.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 452.45 | 455.01 | 459.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 450.10 | 455.01 | 459.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 460.95 | 456.28 | 459.16 | SL hit (close>static) qty=1.00 sl=460.60 alert=retest2 |

### Cycle 51 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 473.95 | 461.24 | 461.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 477.85 | 466.65 | 463.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 499.00 | 499.52 | 489.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 11:30:00 | 497.95 | 499.52 | 489.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 490.30 | 497.84 | 493.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 490.30 | 497.84 | 493.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 489.95 | 496.26 | 493.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:45:00 | 493.90 | 494.23 | 492.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 500.10 | 493.63 | 492.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-18 14:15:00 | 543.29 | 534.69 | 527.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 552.85 | 558.87 | 559.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 551.00 | 555.85 | 557.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 549.75 | 549.45 | 552.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 549.75 | 549.45 | 552.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 549.75 | 549.45 | 552.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:00:00 | 545.20 | 548.60 | 551.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 517.94 | 524.92 | 528.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 490.68 | 505.57 | 515.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 520.20 | 496.46 | 493.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 14:15:00 | 522.50 | 513.58 | 510.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 514.50 | 516.23 | 513.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:00:00 | 514.50 | 516.23 | 513.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 509.40 | 514.87 | 513.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 510.00 | 514.87 | 513.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 508.95 | 513.68 | 512.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 511.55 | 513.68 | 512.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 505.50 | 512.05 | 512.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 505.50 | 512.05 | 512.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 12:15:00 | 504.25 | 507.93 | 509.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 473.50 | 470.19 | 477.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-30 09:45:00 | 471.25 | 470.19 | 477.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 479.40 | 472.03 | 477.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 479.40 | 472.03 | 477.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 482.65 | 474.15 | 477.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:00:00 | 482.65 | 474.15 | 477.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 483.75 | 476.07 | 478.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 485.20 | 476.07 | 478.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 481.80 | 478.49 | 478.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:45:00 | 477.10 | 477.92 | 478.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 482.00 | 475.54 | 475.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 482.00 | 475.54 | 475.21 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 471.60 | 474.75 | 474.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 469.55 | 473.07 | 474.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 470.50 | 469.16 | 471.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 470.50 | 469.16 | 471.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 471.85 | 469.69 | 471.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 474.00 | 469.69 | 471.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 469.30 | 469.62 | 471.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:45:00 | 467.90 | 469.21 | 470.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 474.15 | 470.80 | 471.24 | SL hit (close>static) qty=1.00 sl=472.85 alert=retest2 |

### Cycle 57 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 474.95 | 471.63 | 471.57 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 467.45 | 471.31 | 471.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 464.90 | 470.03 | 470.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 476.50 | 464.03 | 465.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 476.50 | 464.03 | 465.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 476.50 | 464.03 | 465.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 476.25 | 464.03 | 465.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 475.40 | 468.18 | 467.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 478.15 | 471.48 | 469.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 471.85 | 475.34 | 472.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 471.85 | 475.34 | 472.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 471.85 | 475.34 | 472.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 471.85 | 475.34 | 472.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 473.25 | 474.92 | 472.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:30:00 | 474.30 | 474.66 | 472.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 15:15:00 | 474.00 | 474.66 | 472.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:30:00 | 474.80 | 474.33 | 473.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:45:00 | 474.45 | 474.45 | 473.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 474.45 | 474.45 | 473.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 473.30 | 474.45 | 473.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 475.00 | 474.56 | 473.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 475.75 | 474.65 | 473.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 09:15:00 | 521.40 | 502.45 | 492.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 500.60 | 504.68 | 505.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 495.55 | 502.85 | 504.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 501.75 | 500.71 | 502.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 501.75 | 500.71 | 502.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 501.75 | 500.71 | 502.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 501.75 | 500.71 | 502.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 501.05 | 500.77 | 502.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 499.65 | 500.31 | 501.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 506.00 | 500.25 | 501.23 | SL hit (close>static) qty=1.00 sl=503.70 alert=retest2 |

### Cycle 61 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 503.65 | 501.77 | 501.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 506.60 | 502.74 | 502.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 500.70 | 502.95 | 502.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 500.70 | 502.95 | 502.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 500.70 | 502.95 | 502.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 499.50 | 502.95 | 502.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 499.80 | 502.32 | 502.25 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 500.85 | 502.03 | 502.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 497.00 | 501.02 | 501.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 10:15:00 | 497.65 | 496.93 | 498.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 10:15:00 | 497.65 | 496.93 | 498.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 497.65 | 496.93 | 498.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 497.65 | 496.93 | 498.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 498.55 | 497.25 | 498.34 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 501.65 | 499.44 | 499.15 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 493.40 | 498.23 | 498.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 492.55 | 497.10 | 498.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 496.35 | 493.27 | 495.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 496.35 | 493.27 | 495.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 496.35 | 493.27 | 495.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 496.35 | 493.27 | 495.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 495.20 | 493.65 | 495.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:45:00 | 493.50 | 493.63 | 495.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 493.50 | 493.63 | 495.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 498.60 | 494.63 | 495.48 | SL hit (close>static) qty=1.00 sl=497.35 alert=retest2 |

### Cycle 65 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 498.00 | 496.35 | 496.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 506.20 | 498.32 | 497.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 498.35 | 500.07 | 498.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 498.35 | 500.07 | 498.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 498.35 | 500.07 | 498.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 498.35 | 500.07 | 498.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 499.90 | 500.04 | 498.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 502.20 | 500.04 | 498.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 502.30 | 500.49 | 499.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 518.85 | 529.79 | 530.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 518.85 | 529.79 | 530.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 516.05 | 527.04 | 529.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 510.95 | 510.82 | 517.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:30:00 | 511.20 | 510.82 | 517.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 504.90 | 510.51 | 514.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:00:00 | 501.70 | 507.21 | 511.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 501.95 | 505.43 | 508.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 501.05 | 504.03 | 506.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 500.60 | 496.97 | 496.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 500.60 | 496.97 | 496.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 507.00 | 499.35 | 497.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 507.40 | 507.57 | 504.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 507.10 | 507.57 | 504.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 519.70 | 520.50 | 517.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 519.70 | 520.50 | 517.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 521.40 | 523.15 | 521.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 522.10 | 523.15 | 521.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 519.95 | 522.51 | 521.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 519.95 | 522.51 | 521.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 518.80 | 521.77 | 520.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:30:00 | 519.35 | 521.77 | 520.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 517.45 | 520.91 | 520.59 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 518.25 | 520.37 | 520.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 511.85 | 518.25 | 519.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 512.25 | 509.32 | 512.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 512.25 | 509.32 | 512.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 512.25 | 509.32 | 512.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 512.25 | 509.32 | 512.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 512.25 | 509.91 | 512.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 517.95 | 509.91 | 512.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 520.50 | 512.03 | 513.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 520.35 | 512.03 | 513.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 516.15 | 512.85 | 513.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:45:00 | 520.90 | 512.85 | 513.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 511.25 | 510.81 | 511.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 515.20 | 510.81 | 511.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 507.60 | 510.17 | 511.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 506.05 | 510.17 | 511.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:45:00 | 505.00 | 508.44 | 510.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 513.70 | 509.99 | 510.84 | SL hit (close>static) qty=1.00 sl=511.60 alert=retest2 |

### Cycle 69 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 518.35 | 512.30 | 511.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 519.35 | 513.71 | 512.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 514.90 | 515.90 | 514.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 514.90 | 515.90 | 514.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 514.90 | 515.90 | 514.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 514.90 | 515.90 | 514.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 514.00 | 515.52 | 514.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 522.10 | 515.52 | 514.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 12:15:00 | 525.50 | 528.80 | 528.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 525.50 | 528.80 | 528.86 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 530.55 | 529.02 | 528.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 534.15 | 530.68 | 529.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 530.20 | 531.14 | 530.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 530.20 | 531.14 | 530.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 530.20 | 531.14 | 530.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 530.20 | 531.14 | 530.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 531.50 | 531.22 | 530.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:30:00 | 532.00 | 531.29 | 530.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 534.60 | 531.23 | 530.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:00:00 | 533.75 | 531.74 | 530.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 532.50 | 531.60 | 530.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 532.05 | 531.76 | 531.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 532.05 | 531.76 | 531.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 531.60 | 531.73 | 531.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 533.10 | 531.62 | 531.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:45:00 | 532.90 | 531.71 | 531.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:45:00 | 532.60 | 532.03 | 531.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:45:00 | 533.00 | 532.43 | 531.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 531.95 | 532.87 | 532.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 531.95 | 532.87 | 532.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 531.85 | 532.67 | 532.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 532.60 | 532.67 | 532.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 527.85 | 531.70 | 531.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 527.85 | 531.70 | 531.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 524.10 | 530.01 | 530.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 523.00 | 521.95 | 525.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:45:00 | 524.00 | 521.95 | 525.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 524.60 | 522.48 | 525.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:45:00 | 526.45 | 522.48 | 525.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 522.50 | 522.48 | 525.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 13:30:00 | 521.80 | 522.29 | 524.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:15:00 | 519.00 | 522.27 | 524.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 521.75 | 521.64 | 523.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 521.85 | 523.48 | 523.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 528.25 | 524.64 | 524.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 528.25 | 524.64 | 524.24 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 517.65 | 524.28 | 524.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 514.25 | 520.45 | 522.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 10:15:00 | 520.40 | 519.80 | 521.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 10:15:00 | 520.40 | 519.80 | 521.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 520.40 | 519.80 | 521.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:30:00 | 520.50 | 519.80 | 521.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 519.55 | 519.75 | 521.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 12:15:00 | 518.50 | 519.75 | 521.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 13:45:00 | 517.75 | 519.05 | 520.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 527.95 | 519.15 | 520.20 | SL hit (close>static) qty=1.00 sl=522.85 alert=retest2 |

### Cycle 75 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 532.55 | 521.83 | 521.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 11:15:00 | 536.60 | 524.79 | 522.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 524.00 | 534.46 | 532.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 524.00 | 534.46 | 532.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 524.00 | 534.46 | 532.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 524.00 | 534.46 | 532.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 528.30 | 533.23 | 531.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 11:45:00 | 530.50 | 533.28 | 531.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 530.70 | 532.77 | 532.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 524.15 | 531.04 | 531.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 524.15 | 531.04 | 531.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 521.85 | 528.54 | 530.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 533.80 | 528.87 | 530.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 533.80 | 528.87 | 530.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 533.80 | 528.87 | 530.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 533.80 | 528.87 | 530.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 535.20 | 530.13 | 530.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 528.95 | 529.51 | 530.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:00:00 | 530.90 | 526.57 | 527.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 532.95 | 529.06 | 528.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 532.95 | 529.06 | 528.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 535.00 | 530.25 | 529.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 533.65 | 534.35 | 532.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 533.65 | 534.35 | 532.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 532.40 | 533.78 | 532.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 534.25 | 533.78 | 532.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 528.50 | 533.42 | 533.35 | SL hit (close<static) qty=1.00 sl=532.15 alert=retest2 |

### Cycle 78 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 529.80 | 532.70 | 533.03 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 535.70 | 532.87 | 532.69 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 532.40 | 533.26 | 533.31 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 533.75 | 533.41 | 533.37 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 532.95 | 533.39 | 533.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 531.10 | 532.93 | 533.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 521.60 | 519.05 | 522.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 521.60 | 519.05 | 522.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 521.60 | 519.05 | 522.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 521.60 | 519.05 | 522.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 522.75 | 519.79 | 522.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:45:00 | 522.70 | 519.79 | 522.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 524.20 | 520.67 | 522.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 524.20 | 520.67 | 522.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 521.75 | 521.53 | 522.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 514.65 | 521.42 | 522.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 488.92 | 501.54 | 508.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 494.10 | 493.30 | 499.10 | SL hit (close>ema200) qty=0.50 sl=493.30 alert=retest2 |

### Cycle 83 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 507.30 | 500.71 | 500.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 510.20 | 502.61 | 501.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 509.00 | 510.31 | 507.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:45:00 | 509.00 | 510.31 | 507.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 506.70 | 509.27 | 507.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 507.35 | 509.27 | 507.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 505.30 | 508.47 | 507.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 505.30 | 508.47 | 507.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 502.00 | 506.10 | 506.62 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 512.90 | 505.93 | 505.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 515.75 | 507.89 | 506.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 510.30 | 510.40 | 508.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 510.30 | 510.40 | 508.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 508.00 | 509.87 | 508.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 508.20 | 509.87 | 508.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 507.40 | 509.38 | 508.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 507.40 | 509.38 | 508.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 506.80 | 508.55 | 508.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 506.40 | 508.55 | 508.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 505.70 | 507.72 | 507.87 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 517.85 | 509.63 | 508.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 524.40 | 518.12 | 514.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 521.75 | 521.88 | 517.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 521.75 | 521.88 | 517.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 536.10 | 537.97 | 535.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 536.00 | 537.97 | 535.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 535.10 | 537.39 | 535.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 535.10 | 537.39 | 535.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 534.50 | 536.81 | 535.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 540.30 | 536.53 | 535.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 537.40 | 536.56 | 535.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 534.80 | 542.66 | 543.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 534.80 | 542.66 | 543.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 527.20 | 536.50 | 539.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 528.00 | 525.04 | 530.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 528.00 | 525.04 | 530.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 528.00 | 525.04 | 530.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 528.00 | 525.04 | 530.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 528.20 | 525.67 | 529.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 526.00 | 525.67 | 529.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:00:00 | 527.40 | 526.36 | 529.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 527.30 | 527.17 | 528.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 527.50 | 527.33 | 528.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 531.00 | 528.07 | 529.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 532.90 | 529.28 | 529.30 | SL hit (close>static) qty=1.00 sl=531.85 alert=retest2 |

### Cycle 89 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 531.90 | 529.80 | 529.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 535.95 | 532.31 | 530.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 533.05 | 533.66 | 532.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 10:30:00 | 532.30 | 533.66 | 532.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 537.80 | 544.55 | 541.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 537.80 | 544.55 | 541.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 532.65 | 542.17 | 541.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 532.65 | 542.17 | 541.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 535.20 | 539.70 | 540.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 533.45 | 537.76 | 539.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 539.60 | 537.50 | 538.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 539.60 | 537.50 | 538.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 539.60 | 537.50 | 538.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 539.60 | 537.50 | 538.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 539.60 | 537.92 | 538.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 539.00 | 537.92 | 538.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 542.25 | 538.79 | 538.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 542.25 | 538.79 | 538.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 542.00 | 539.43 | 539.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 543.05 | 540.92 | 540.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 542.65 | 543.77 | 542.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 542.65 | 543.77 | 542.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 542.65 | 543.77 | 542.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 541.80 | 543.77 | 542.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 541.40 | 543.30 | 542.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 540.55 | 543.30 | 542.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 544.30 | 543.50 | 542.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 545.15 | 543.72 | 542.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:15:00 | 545.05 | 543.72 | 542.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 545.70 | 544.12 | 542.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:45:00 | 545.30 | 543.96 | 542.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 545.30 | 544.23 | 543.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 549.50 | 544.23 | 543.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 541.45 | 543.67 | 542.95 | SL hit (close<static) qty=1.00 sl=542.15 alert=retest2 |

### Cycle 92 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 536.45 | 541.56 | 542.07 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 545.45 | 542.05 | 541.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 549.60 | 545.82 | 544.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 544.60 | 547.08 | 545.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 544.60 | 547.08 | 545.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 544.60 | 547.08 | 545.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 543.60 | 547.08 | 545.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 544.55 | 546.58 | 545.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 542.70 | 546.58 | 545.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 545.80 | 546.42 | 545.53 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 541.35 | 544.54 | 544.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 522.25 | 539.03 | 542.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 11:15:00 | 529.00 | 528.88 | 532.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 12:15:00 | 530.15 | 528.88 | 532.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 532.80 | 529.76 | 532.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 532.80 | 529.76 | 532.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 534.50 | 530.71 | 532.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 537.80 | 530.71 | 532.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 530.60 | 531.28 | 532.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:45:00 | 527.90 | 530.57 | 531.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 536.50 | 531.76 | 531.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 536.50 | 531.76 | 531.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 537.60 | 534.31 | 533.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 532.15 | 537.31 | 535.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 14:15:00 | 532.15 | 537.31 | 535.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 532.15 | 537.31 | 535.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 532.15 | 537.31 | 535.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 536.90 | 537.23 | 535.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 533.00 | 537.23 | 535.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 531.00 | 535.98 | 535.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 530.70 | 535.98 | 535.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 533.05 | 535.40 | 535.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 530.00 | 535.40 | 535.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 536.25 | 535.56 | 535.24 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 532.00 | 535.34 | 535.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 528.00 | 533.87 | 534.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 533.95 | 531.37 | 532.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 12:15:00 | 533.95 | 531.37 | 532.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 533.95 | 531.37 | 532.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 533.95 | 531.37 | 532.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 534.25 | 531.95 | 533.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 535.70 | 531.95 | 533.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 535.00 | 532.55 | 533.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 531.35 | 532.55 | 533.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 529.40 | 531.92 | 532.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:00:00 | 528.20 | 530.70 | 531.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:45:00 | 527.60 | 529.91 | 531.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 526.95 | 528.34 | 530.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 526.55 | 523.56 | 523.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 526.55 | 523.56 | 523.32 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 519.35 | 522.61 | 522.93 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 525.45 | 523.58 | 523.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 527.00 | 524.27 | 523.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 525.60 | 525.84 | 524.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 525.60 | 525.84 | 524.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 524.15 | 525.50 | 524.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 524.15 | 525.50 | 524.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 525.20 | 525.44 | 524.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 523.60 | 525.44 | 524.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 526.00 | 525.55 | 524.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:45:00 | 529.05 | 527.52 | 526.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 529.30 | 528.22 | 526.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 529.05 | 528.33 | 526.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 529.05 | 528.23 | 527.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 528.50 | 528.29 | 527.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 531.95 | 529.64 | 528.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 521.15 | 527.10 | 527.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 521.15 | 527.10 | 527.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 520.35 | 524.70 | 525.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 481.65 | 479.33 | 484.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:45:00 | 482.35 | 479.33 | 484.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 483.10 | 480.42 | 484.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 483.00 | 480.42 | 484.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 486.10 | 481.56 | 484.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 486.10 | 481.56 | 484.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 486.60 | 482.57 | 484.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 486.60 | 482.57 | 484.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 483.75 | 484.89 | 485.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 481.55 | 484.89 | 485.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 13:15:00 | 487.40 | 485.40 | 485.46 | SL hit (close>static) qty=1.00 sl=486.75 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 14:15:00 | 487.85 | 485.89 | 485.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 13:15:00 | 488.40 | 486.73 | 486.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 486.55 | 487.41 | 486.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 10:15:00 | 486.55 | 487.41 | 486.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 486.55 | 487.41 | 486.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 486.55 | 487.41 | 486.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 484.35 | 486.80 | 486.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 484.35 | 486.80 | 486.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 489.35 | 487.31 | 486.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 492.20 | 488.29 | 487.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:00:00 | 492.25 | 489.08 | 487.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 11:15:00 | 482.90 | 487.58 | 487.48 | SL hit (close<static) qty=1.00 sl=484.10 alert=retest2 |

### Cycle 102 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 484.00 | 486.86 | 487.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 480.55 | 485.60 | 486.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 462.65 | 458.04 | 464.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 462.65 | 458.04 | 464.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 462.65 | 458.04 | 464.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 462.65 | 458.04 | 464.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 463.20 | 459.07 | 464.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 467.70 | 459.07 | 464.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 463.85 | 460.02 | 464.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 469.20 | 460.02 | 464.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 460.95 | 460.21 | 464.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 462.90 | 460.21 | 464.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 460.40 | 460.25 | 463.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 461.80 | 460.25 | 463.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 461.40 | 454.84 | 457.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 464.45 | 454.84 | 457.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 455.75 | 455.02 | 456.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 454.25 | 455.02 | 456.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 454.00 | 455.35 | 456.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 454.20 | 455.12 | 456.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 458.00 | 456.44 | 456.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 458.00 | 456.44 | 456.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 461.30 | 457.77 | 457.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 479.90 | 481.77 | 474.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 479.90 | 481.77 | 474.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 475.90 | 479.59 | 475.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 475.90 | 479.59 | 475.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 477.00 | 479.08 | 475.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 478.50 | 479.14 | 475.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:00:00 | 479.60 | 479.29 | 476.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 469.10 | 475.43 | 475.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 469.10 | 475.43 | 475.92 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 477.00 | 474.36 | 474.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 481.05 | 476.09 | 474.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 478.00 | 478.36 | 476.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 479.35 | 478.36 | 476.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 483.30 | 483.73 | 482.12 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 479.15 | 481.36 | 481.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 476.45 | 480.38 | 481.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 476.10 | 472.04 | 474.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 476.10 | 472.04 | 474.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 476.10 | 472.04 | 474.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 474.00 | 472.04 | 474.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 477.80 | 473.19 | 474.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 477.80 | 473.19 | 474.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 482.50 | 476.46 | 475.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 495.25 | 482.67 | 479.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 515.95 | 516.18 | 509.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 515.95 | 516.18 | 509.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 509.50 | 513.82 | 511.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 509.50 | 513.82 | 511.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 512.40 | 513.54 | 511.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 510.90 | 513.54 | 511.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 512.20 | 513.27 | 511.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 511.50 | 513.27 | 511.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 512.40 | 513.10 | 511.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 511.40 | 513.10 | 511.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 514.00 | 513.28 | 511.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 512.90 | 513.28 | 511.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 511.85 | 512.91 | 511.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 509.80 | 512.91 | 511.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 510.50 | 512.43 | 511.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 510.50 | 512.43 | 511.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 507.45 | 511.43 | 511.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 507.45 | 511.43 | 511.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 504.85 | 510.11 | 510.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 501.05 | 507.16 | 509.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 495.30 | 492.04 | 496.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 495.30 | 492.04 | 496.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 499.40 | 493.84 | 496.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 499.40 | 493.84 | 496.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 496.00 | 494.27 | 496.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 493.20 | 494.27 | 496.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 491.95 | 493.80 | 496.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 488.40 | 491.53 | 494.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 497.80 | 495.60 | 495.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 497.80 | 495.60 | 495.30 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 492.00 | 495.14 | 495.24 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 498.90 | 495.22 | 495.19 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 488.75 | 493.93 | 494.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 487.45 | 492.63 | 493.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 478.55 | 478.31 | 482.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 486.25 | 478.31 | 482.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 488.80 | 480.41 | 482.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 490.40 | 480.41 | 482.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 487.55 | 481.84 | 483.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 486.20 | 481.84 | 483.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 491.55 | 485.06 | 484.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 491.55 | 485.06 | 484.45 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 479.90 | 485.26 | 485.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 477.20 | 483.65 | 484.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 15:15:00 | 482.20 | 482.02 | 483.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-27 09:15:00 | 449.60 | 482.02 | 483.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 449.45 | 447.73 | 454.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 452.55 | 447.73 | 454.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 454.00 | 449.56 | 454.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 454.65 | 450.58 | 454.10 | SL hit (close>ema400) qty=1.00 sl=454.10 alert=retest1 |

### Cycle 115 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 457.55 | 455.43 | 455.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 459.75 | 456.29 | 455.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 456.60 | 456.63 | 456.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 456.60 | 456.63 | 456.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 456.60 | 456.63 | 456.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 453.20 | 456.63 | 456.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 459.90 | 457.28 | 456.37 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 452.25 | 455.67 | 455.76 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 456.75 | 451.08 | 451.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 462.95 | 453.45 | 452.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 464.10 | 464.66 | 460.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 464.10 | 464.66 | 460.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 468.40 | 468.86 | 465.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 466.60 | 468.86 | 465.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 478.10 | 480.33 | 478.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 478.10 | 480.33 | 478.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 479.95 | 480.25 | 478.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:15:00 | 480.30 | 480.25 | 478.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 477.00 | 479.44 | 478.75 | SL hit (close<static) qty=1.00 sl=477.65 alert=retest2 |

### Cycle 118 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 471.20 | 477.33 | 477.94 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 485.70 | 478.34 | 477.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 487.15 | 482.00 | 479.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 10:15:00 | 484.35 | 486.34 | 484.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 10:15:00 | 484.35 | 486.34 | 484.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 484.35 | 486.34 | 484.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 484.35 | 486.34 | 484.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 485.30 | 486.13 | 484.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:30:00 | 484.75 | 486.13 | 484.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 485.45 | 485.74 | 484.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 486.20 | 485.83 | 484.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 493.10 | 485.69 | 484.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 487.30 | 486.86 | 485.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 486.95 | 486.52 | 485.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 485.40 | 486.30 | 485.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 485.10 | 486.30 | 485.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 483.25 | 485.69 | 485.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-19 13:15:00 | 483.25 | 485.69 | 485.24 | SL hit (close<static) qty=1.00 sl=484.45 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 479.65 | 484.48 | 484.74 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 491.35 | 484.98 | 484.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 492.25 | 487.38 | 485.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 489.80 | 490.06 | 487.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:00:00 | 489.80 | 490.06 | 487.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 491.50 | 490.35 | 488.22 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 484.70 | 487.13 | 487.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 482.70 | 485.63 | 486.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 489.45 | 485.77 | 486.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 489.45 | 485.77 | 486.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 489.45 | 485.77 | 486.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 489.45 | 485.77 | 486.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 490.50 | 486.72 | 486.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 491.65 | 486.72 | 486.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 489.95 | 487.37 | 487.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 500.20 | 491.67 | 489.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 488.80 | 495.40 | 493.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 488.80 | 495.40 | 493.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 488.80 | 495.40 | 493.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 488.80 | 495.40 | 493.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 491.55 | 494.63 | 493.10 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 13:15:00 | 490.70 | 492.12 | 492.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 488.30 | 491.35 | 491.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 483.00 | 482.92 | 486.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 465.90 | 482.92 | 486.25 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 476.05 | 473.46 | 477.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:15:00 | 478.60 | 473.46 | 477.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 477.05 | 474.17 | 477.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 477.05 | 474.17 | 477.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 474.05 | 474.15 | 476.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 478.50 | 475.02 | 477.03 | SL hit (close>ema400) qty=1.00 sl=477.03 alert=retest1 |

### Cycle 125 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 490.70 | 479.25 | 478.67 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 475.95 | 481.33 | 481.79 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 489.50 | 482.44 | 481.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 493.45 | 487.81 | 485.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 490.35 | 490.49 | 487.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 490.35 | 490.49 | 487.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 486.95 | 489.78 | 487.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 486.95 | 489.78 | 487.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 487.65 | 489.36 | 487.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 491.00 | 489.36 | 487.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 495.10 | 490.50 | 488.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 504.00 | 490.50 | 488.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:15:00 | 500.55 | 508.88 | 506.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 12:15:00 | 498.70 | 505.26 | 505.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 498.70 | 505.26 | 505.33 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 510.05 | 505.99 | 505.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 517.75 | 509.30 | 508.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 507.10 | 510.81 | 509.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 507.10 | 510.81 | 509.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 507.10 | 510.81 | 509.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 507.10 | 510.81 | 509.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 506.00 | 509.85 | 509.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 497.80 | 509.85 | 509.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 489.10 | 505.70 | 507.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 488.45 | 502.25 | 505.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 487.55 | 487.42 | 493.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 489.15 | 487.42 | 493.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 501.90 | 489.33 | 492.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 501.90 | 489.33 | 492.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 502.30 | 491.92 | 493.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 502.30 | 491.92 | 493.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 497.00 | 494.74 | 494.46 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 491.60 | 493.95 | 494.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 484.05 | 489.78 | 491.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 482.00 | 477.17 | 482.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 482.00 | 477.17 | 482.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 482.00 | 477.17 | 482.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 482.90 | 477.17 | 482.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 478.80 | 477.50 | 481.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 479.45 | 477.50 | 481.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 486.35 | 479.27 | 482.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 486.35 | 479.27 | 482.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 488.35 | 481.09 | 482.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 488.50 | 481.09 | 482.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 483.55 | 482.20 | 483.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:30:00 | 482.15 | 482.20 | 483.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 484.35 | 482.63 | 483.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 467.10 | 482.63 | 483.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 489.45 | 480.46 | 481.32 | SL hit (close>static) qty=1.00 sl=485.25 alert=retest2 |

### Cycle 133 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 491.85 | 482.73 | 482.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 494.85 | 488.66 | 485.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 487.90 | 490.64 | 487.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 487.90 | 490.64 | 487.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 487.90 | 490.64 | 487.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 488.00 | 490.64 | 487.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 487.20 | 489.95 | 487.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 487.20 | 489.95 | 487.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 486.70 | 489.30 | 487.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 486.70 | 489.30 | 487.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 484.60 | 488.36 | 487.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 483.75 | 488.36 | 487.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 488.90 | 488.32 | 487.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:15:00 | 486.80 | 488.32 | 487.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 486.80 | 488.01 | 487.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 503.00 | 488.01 | 487.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 15:15:00 | 489.00 | 493.47 | 493.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 489.00 | 493.47 | 493.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 15:15:00 | 487.95 | 491.04 | 492.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 09:15:00 | 493.95 | 491.63 | 492.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 493.95 | 491.63 | 492.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 493.95 | 491.63 | 492.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:00:00 | 493.95 | 491.63 | 492.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 497.50 | 492.80 | 492.92 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 11:15:00 | 504.15 | 495.07 | 493.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 12:15:00 | 510.00 | 498.06 | 495.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 525.45 | 525.47 | 516.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 12:00:00 | 525.45 | 525.47 | 516.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 535.10 | 535.75 | 529.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 531.00 | 535.75 | 529.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 546.40 | 547.05 | 542.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 553.30 | 547.05 | 542.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 539.20 | 554.64 | 553.80 | SL hit (close<static) qty=1.00 sl=540.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 544.65 | 552.64 | 552.97 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 569.25 | 552.90 | 552.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 574.35 | 559.69 | 555.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 09:15:00 | 577.00 | 577.52 | 570.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 11:15:00 | 567.80 | 574.70 | 570.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 567.80 | 574.70 | 570.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 567.80 | 574.70 | 570.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 567.40 | 573.24 | 570.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 567.40 | 573.24 | 570.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 558.00 | 567.15 | 567.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 553.05 | 564.33 | 566.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 560.50 | 560.50 | 563.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 562.10 | 560.50 | 563.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 556.60 | 559.72 | 562.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 555.50 | 559.72 | 562.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 555.00 | 558.30 | 561.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 564.50 | 562.39 | 562.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 564.50 | 562.39 | 562.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 567.25 | 563.46 | 562.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 15:15:00 | 566.15 | 566.25 | 564.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:15:00 | 572.85 | 566.25 | 564.74 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 566.85 | 572.29 | 569.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 566.85 | 572.29 | 569.92 | SL hit (close<ema400) qty=1.00 sl=569.92 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 607.70 | 2024-05-17 15:15:00 | 590.00 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-05-17 10:00:00 | 605.80 | 2024-05-17 15:15:00 | 590.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-05-17 13:15:00 | 600.20 | 2024-05-17 15:15:00 | 590.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-21 09:15:00 | 614.10 | 2024-05-24 15:15:00 | 597.30 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-05-23 09:15:00 | 621.50 | 2024-05-24 15:15:00 | 597.30 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-05-23 14:30:00 | 618.90 | 2024-05-24 15:15:00 | 597.30 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2024-05-24 09:15:00 | 623.40 | 2024-05-24 15:15:00 | 597.30 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-05-24 12:30:00 | 618.00 | 2024-05-24 15:15:00 | 597.30 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2024-06-03 09:15:00 | 643.00 | 2024-06-04 10:15:00 | 584.30 | STOP_HIT | 1.00 | -9.13% |
| BUY | retest2 | 2024-07-01 09:15:00 | 739.50 | 2024-07-02 10:15:00 | 729.90 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-04 11:30:00 | 726.30 | 2024-07-05 14:15:00 | 737.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-07-04 14:15:00 | 726.55 | 2024-07-05 14:15:00 | 737.30 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-07-12 09:15:00 | 713.00 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2024-07-12 09:45:00 | 711.45 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-07-12 10:30:00 | 711.45 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-07-12 11:30:00 | 712.65 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2024-07-18 09:45:00 | 699.15 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-07-18 11:00:00 | 699.25 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-07-18 11:30:00 | 698.80 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-19 12:30:00 | 698.20 | 2024-07-23 09:15:00 | 711.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-08-26 09:15:00 | 710.20 | 2024-08-29 13:15:00 | 718.60 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2024-08-26 11:15:00 | 705.10 | 2024-08-29 13:15:00 | 718.60 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2024-09-02 12:30:00 | 712.05 | 2024-09-04 09:15:00 | 676.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 12:30:00 | 712.05 | 2024-09-04 13:15:00 | 692.90 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2024-10-17 12:00:00 | 685.50 | 2024-10-23 09:15:00 | 651.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:45:00 | 683.90 | 2024-10-23 09:15:00 | 649.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:15:00 | 682.50 | 2024-10-23 09:15:00 | 651.08 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2024-10-17 12:00:00 | 685.50 | 2024-10-23 13:15:00 | 669.00 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2024-10-21 09:45:00 | 683.90 | 2024-10-23 13:15:00 | 669.00 | STOP_HIT | 0.50 | 2.18% |
| SELL | retest2 | 2024-10-21 10:15:00 | 682.50 | 2024-10-23 13:15:00 | 669.00 | STOP_HIT | 0.50 | 1.98% |
| SELL | retest2 | 2024-10-21 11:00:00 | 685.35 | 2024-10-23 14:15:00 | 688.05 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-10-22 12:15:00 | 673.60 | 2024-10-23 14:15:00 | 688.05 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-23 11:45:00 | 662.50 | 2024-10-24 09:15:00 | 682.45 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-10-28 09:15:00 | 665.00 | 2024-10-29 13:15:00 | 677.70 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-10-28 11:00:00 | 665.30 | 2024-10-29 13:15:00 | 677.70 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-10-28 11:30:00 | 664.90 | 2024-10-29 13:15:00 | 677.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-10-29 11:45:00 | 663.85 | 2024-10-29 13:15:00 | 677.70 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-11-04 11:15:00 | 678.90 | 2024-11-04 14:15:00 | 662.50 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-11-25 15:00:00 | 674.30 | 2024-12-04 11:15:00 | 640.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-25 15:00:00 | 674.30 | 2024-12-04 12:15:00 | 647.10 | STOP_HIT | 0.50 | 4.03% |
| BUY | retest2 | 2024-12-09 15:15:00 | 676.10 | 2024-12-10 10:15:00 | 665.80 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-11 09:15:00 | 677.85 | 2024-12-13 09:15:00 | 670.05 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-12-11 10:15:00 | 676.15 | 2024-12-18 10:15:00 | 681.25 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2024-12-11 11:00:00 | 675.80 | 2024-12-18 10:15:00 | 681.25 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-12-12 11:30:00 | 678.60 | 2024-12-18 10:15:00 | 681.25 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-12-13 12:15:00 | 678.45 | 2024-12-18 10:15:00 | 681.25 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-12-30 10:45:00 | 651.00 | 2025-01-01 09:15:00 | 649.70 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-01-14 12:45:00 | 544.70 | 2025-01-15 09:15:00 | 559.20 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-01-20 11:45:00 | 570.45 | 2025-01-21 12:15:00 | 561.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-01-23 13:15:00 | 550.70 | 2025-01-28 09:15:00 | 523.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:00:00 | 550.10 | 2025-01-28 09:15:00 | 522.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 550.15 | 2025-01-28 09:15:00 | 522.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 550.25 | 2025-01-28 09:15:00 | 522.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:15:00 | 550.70 | 2025-01-29 09:15:00 | 495.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 14:00:00 | 550.10 | 2025-01-29 09:15:00 | 495.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 550.15 | 2025-01-29 09:15:00 | 495.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 550.25 | 2025-01-29 09:15:00 | 495.23 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-07 11:15:00 | 492.30 | 2025-02-10 09:15:00 | 477.35 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-02-14 09:15:00 | 459.05 | 2025-02-14 13:15:00 | 436.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 459.05 | 2025-02-17 14:15:00 | 432.50 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2025-03-03 11:15:00 | 450.10 | 2025-03-03 12:15:00 | 460.95 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-03-07 14:45:00 | 493.90 | 2025-03-18 14:15:00 | 543.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-10 09:15:00 | 500.10 | 2025-03-18 14:15:00 | 550.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 11:00:00 | 545.20 | 2025-04-04 09:15:00 | 517.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 11:00:00 | 545.20 | 2025-04-07 09:15:00 | 490.68 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-23 09:15:00 | 511.55 | 2025-04-23 09:15:00 | 505.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-05-02 10:45:00 | 477.10 | 2025-05-05 15:15:00 | 482.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-07 14:45:00 | 467.90 | 2025-05-08 10:15:00 | 474.15 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-05-13 14:30:00 | 474.30 | 2025-05-19 09:15:00 | 521.40 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-05-13 15:15:00 | 474.00 | 2025-05-19 12:15:00 | 521.73 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2025-05-14 10:30:00 | 474.80 | 2025-05-19 12:15:00 | 522.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 11:45:00 | 474.45 | 2025-05-19 12:15:00 | 521.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 15:15:00 | 475.75 | 2025-05-19 12:15:00 | 523.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-23 13:30:00 | 499.65 | 2025-05-26 09:15:00 | 506.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-02 11:45:00 | 493.50 | 2025-06-02 12:15:00 | 498.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-06-02 12:15:00 | 493.50 | 2025-06-02 12:15:00 | 498.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-06-04 09:15:00 | 502.20 | 2025-06-12 13:15:00 | 518.85 | STOP_HIT | 1.00 | 3.32% |
| BUY | retest2 | 2025-06-04 10:00:00 | 502.30 | 2025-06-12 13:15:00 | 518.85 | STOP_HIT | 1.00 | 3.29% |
| SELL | retest2 | 2025-06-17 13:00:00 | 501.70 | 2025-06-23 14:15:00 | 500.60 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-06-18 14:15:00 | 501.95 | 2025-06-23 14:15:00 | 500.60 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-06-19 10:00:00 | 501.05 | 2025-06-23 14:15:00 | 500.60 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-07-08 11:15:00 | 506.05 | 2025-07-08 14:15:00 | 513.70 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-07-08 12:45:00 | 505.00 | 2025-07-08 14:15:00 | 513.70 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-07-10 09:15:00 | 522.10 | 2025-07-18 12:15:00 | 525.50 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-07-21 14:30:00 | 532.00 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-22 09:15:00 | 534.60 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-22 10:00:00 | 533.75 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-22 12:00:00 | 532.50 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-23 09:15:00 | 533.10 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-23 09:45:00 | 532.90 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-23 10:45:00 | 532.60 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-23 11:45:00 | 533.00 | 2025-07-24 11:15:00 | 527.85 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-28 13:30:00 | 521.80 | 2025-07-30 10:15:00 | 528.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-07-28 15:15:00 | 519.00 | 2025-07-30 10:15:00 | 528.25 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-29 10:00:00 | 521.75 | 2025-07-30 10:15:00 | 528.25 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-07-30 09:15:00 | 521.85 | 2025-07-30 10:15:00 | 528.25 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-01 12:15:00 | 518.50 | 2025-08-04 09:15:00 | 527.95 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-08-01 13:45:00 | 517.75 | 2025-08-04 09:15:00 | 527.95 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-08-06 11:45:00 | 530.50 | 2025-08-07 10:15:00 | 524.15 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-08-07 09:30:00 | 530.70 | 2025-08-07 10:15:00 | 524.15 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-08 09:30:00 | 528.95 | 2025-08-11 13:15:00 | 532.95 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-11 11:00:00 | 530.90 | 2025-08-11 13:15:00 | 532.95 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-08-13 09:15:00 | 534.25 | 2025-08-14 10:15:00 | 528.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-26 09:15:00 | 514.65 | 2025-08-29 09:15:00 | 488.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 514.65 | 2025-09-01 11:15:00 | 494.10 | STOP_HIT | 0.50 | 3.99% |
| BUY | retest2 | 2025-09-18 09:15:00 | 540.30 | 2025-09-24 14:15:00 | 534.80 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-09-18 13:15:00 | 537.40 | 2025-09-24 14:15:00 | 534.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-29 11:15:00 | 526.00 | 2025-09-30 15:15:00 | 532.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-29 14:00:00 | 527.40 | 2025-09-30 15:15:00 | 532.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-30 10:00:00 | 527.30 | 2025-09-30 15:15:00 | 532.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-30 10:30:00 | 527.50 | 2025-09-30 15:15:00 | 532.90 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-13 12:45:00 | 545.15 | 2025-10-14 09:15:00 | 541.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-13 13:15:00 | 545.05 | 2025-10-14 10:15:00 | 539.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-13 14:00:00 | 545.70 | 2025-10-14 10:15:00 | 539.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-13 14:45:00 | 545.30 | 2025-10-14 10:15:00 | 539.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-14 09:15:00 | 549.50 | 2025-10-14 10:15:00 | 539.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-10-28 10:45:00 | 527.90 | 2025-10-28 11:15:00 | 536.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-11-04 14:00:00 | 528.20 | 2025-11-10 13:15:00 | 526.55 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-11-04 14:45:00 | 527.60 | 2025-11-10 13:15:00 | 526.55 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-11-06 09:30:00 | 526.95 | 2025-11-10 13:15:00 | 526.55 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-11-13 11:45:00 | 529.05 | 2025-11-18 09:15:00 | 521.15 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-11-13 12:30:00 | 529.30 | 2025-11-18 09:15:00 | 521.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-13 14:15:00 | 529.05 | 2025-11-18 09:15:00 | 521.15 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-11-14 09:15:00 | 529.05 | 2025-11-18 09:15:00 | 521.15 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-11-17 09:30:00 | 531.95 | 2025-11-18 09:15:00 | 521.15 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-11-27 12:15:00 | 481.55 | 2025-11-27 13:15:00 | 487.40 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-12-01 14:00:00 | 492.20 | 2025-12-02 11:15:00 | 482.90 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-01 15:00:00 | 492.25 | 2025-12-02 11:15:00 | 482.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-12-10 11:15:00 | 454.25 | 2025-12-11 13:15:00 | 458.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-10 12:30:00 | 454.00 | 2025-12-11 13:15:00 | 458.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-10 14:00:00 | 454.20 | 2025-12-11 13:15:00 | 458.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-16 14:45:00 | 478.50 | 2025-12-18 09:15:00 | 469.10 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-12-17 11:00:00 | 479.60 | 2025-12-18 09:15:00 | 469.10 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-01-13 14:15:00 | 488.40 | 2026-01-16 09:15:00 | 497.80 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-01-22 11:15:00 | 486.20 | 2026-01-22 12:15:00 | 491.55 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest1 | 2026-01-27 09:15:00 | 449.60 | 2026-01-29 12:15:00 | 454.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-29 13:45:00 | 451.30 | 2026-01-29 14:15:00 | 459.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-02-11 14:15:00 | 480.30 | 2026-02-12 11:15:00 | 477.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-02-18 14:45:00 | 486.20 | 2026-02-19 13:15:00 | 483.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-02-19 09:15:00 | 493.10 | 2026-02-19 13:15:00 | 483.25 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-19 10:45:00 | 487.30 | 2026-02-19 13:15:00 | 483.25 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-02-19 12:15:00 | 486.95 | 2026-02-19 13:15:00 | 483.25 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2026-03-04 09:15:00 | 465.90 | 2026-03-05 14:15:00 | 478.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-03-12 10:15:00 | 504.00 | 2026-03-16 12:15:00 | 498.70 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-03-16 11:15:00 | 500.55 | 2026-03-16 12:15:00 | 498.70 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-04-02 09:15:00 | 467.10 | 2026-04-02 13:15:00 | 489.45 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2026-04-08 09:15:00 | 503.00 | 2026-04-09 15:15:00 | 489.00 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-04-22 10:15:00 | 553.30 | 2026-04-24 10:15:00 | 539.20 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-05-04 10:15:00 | 555.50 | 2026-05-05 12:15:00 | 564.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-05-04 12:30:00 | 555.00 | 2026-05-05 12:15:00 | 564.50 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest1 | 2026-05-07 09:15:00 | 572.85 | 2026-05-08 10:15:00 | 566.85 | STOP_HIT | 1.00 | -1.05% |
