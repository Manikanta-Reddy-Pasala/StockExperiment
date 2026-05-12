# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 670.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 163 |
| ALERT1 | 106 |
| ALERT2 | 103 |
| ALERT2_SKIP | 64 |
| ALERT3 | 239 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 100 |
| PARTIAL | 20 |
| TARGET_HIT | 2 |
| STOP_HIT | 102 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 62
- **Target hits / Stop hits / Partials:** 2 / 102 / 20
- **Avg / median % per leg:** 1.01% / 0.03%
- **Sum % (uncompounded):** 125.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 12 | 24.0% | 0 | 50 | 0 | -1.01% | -50.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 50 | 12 | 24.0% | 0 | 50 | 0 | -1.01% | -50.3% |
| SELL (all) | 74 | 50 | 67.6% | 2 | 52 | 20 | 2.38% | 175.9% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.98% | 31.8% |
| SELL @ 3rd Alert (retest2) | 66 | 42 | 63.6% | 2 | 48 | 16 | 2.18% | 144.1% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.98% | 31.8% |
| retest2 (combined) | 116 | 54 | 46.6% | 2 | 98 | 16 | 0.81% | 93.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 12:15:00 | 520.05 | 522.49 | 522.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 13:15:00 | 517.60 | 521.51 | 522.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 14:15:00 | 521.80 | 519.67 | 520.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 14:15:00 | 521.80 | 519.67 | 520.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 521.80 | 519.67 | 520.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 521.80 | 519.67 | 520.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 517.95 | 519.32 | 520.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 534.00 | 519.32 | 520.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 547.35 | 524.93 | 522.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 10:15:00 | 551.25 | 530.19 | 525.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 545.15 | 545.88 | 541.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 11:30:00 | 547.00 | 545.88 | 541.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 542.00 | 544.93 | 541.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 541.05 | 544.93 | 541.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 539.40 | 543.83 | 541.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 539.40 | 543.83 | 541.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 539.20 | 542.90 | 541.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 539.20 | 542.90 | 541.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 534.50 | 539.59 | 539.93 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 566.40 | 542.37 | 539.96 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 549.45 | 551.45 | 551.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 544.00 | 549.09 | 550.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 14:15:00 | 545.05 | 544.59 | 547.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 15:00:00 | 545.05 | 544.59 | 547.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 561.00 | 547.94 | 548.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:15:00 | 571.50 | 547.94 | 548.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 585.35 | 555.42 | 551.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 597.40 | 572.26 | 561.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 10:15:00 | 579.80 | 580.63 | 569.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 10:45:00 | 580.40 | 580.63 | 569.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 572.35 | 576.80 | 572.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 575.95 | 576.80 | 572.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 551.05 | 568.96 | 569.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 551.05 | 568.96 | 569.26 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 12:15:00 | 571.65 | 569.50 | 569.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 577.50 | 571.10 | 570.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 682.30 | 683.36 | 663.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:45:00 | 684.55 | 683.36 | 663.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 722.05 | 728.67 | 723.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 720.25 | 728.67 | 723.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 721.25 | 727.18 | 723.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:00:00 | 721.25 | 727.18 | 723.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 727.95 | 727.34 | 723.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 732.20 | 724.40 | 723.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 13:15:00 | 718.70 | 724.20 | 723.82 | SL hit (close<static) qty=1.00 sl=721.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 720.80 | 723.52 | 723.54 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 725.80 | 723.52 | 723.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 11:15:00 | 727.35 | 724.29 | 723.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 721.20 | 725.48 | 724.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 721.20 | 725.48 | 724.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 721.20 | 725.48 | 724.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 717.85 | 725.48 | 724.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 722.00 | 724.79 | 724.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:15:00 | 722.80 | 724.79 | 724.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 11:15:00 | 721.05 | 724.04 | 724.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 721.05 | 724.04 | 724.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 13:15:00 | 719.60 | 722.71 | 723.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 14:15:00 | 725.65 | 723.29 | 723.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 14:15:00 | 725.65 | 723.29 | 723.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 725.65 | 723.29 | 723.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 725.65 | 723.29 | 723.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 723.00 | 723.24 | 723.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 721.45 | 723.24 | 723.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 720.35 | 723.33 | 723.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:00:00 | 721.55 | 720.63 | 721.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 728.95 | 723.33 | 722.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 728.95 | 723.33 | 722.74 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 719.25 | 723.13 | 723.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 716.60 | 719.20 | 720.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 723.30 | 719.35 | 720.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 723.30 | 719.35 | 720.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 723.30 | 719.35 | 720.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 724.40 | 719.35 | 720.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 725.00 | 720.48 | 721.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 725.00 | 720.48 | 721.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 722.10 | 721.45 | 721.39 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 720.10 | 721.18 | 721.27 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 724.10 | 721.41 | 721.32 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 714.25 | 723.06 | 723.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 13:15:00 | 712.00 | 718.50 | 721.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 09:15:00 | 719.80 | 717.66 | 719.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 719.80 | 717.66 | 719.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 719.80 | 717.66 | 719.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 10:30:00 | 714.70 | 717.10 | 719.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 11:00:00 | 714.85 | 717.10 | 719.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 702.80 | 698.94 | 698.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 11:15:00 | 702.80 | 698.94 | 698.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 704.40 | 700.04 | 698.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 15:15:00 | 708.00 | 708.11 | 704.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:15:00 | 708.65 | 708.11 | 704.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 704.70 | 707.43 | 704.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 704.55 | 707.43 | 704.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 711.90 | 708.33 | 705.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 718.55 | 708.38 | 707.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 09:15:00 | 703.25 | 707.35 | 707.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 703.25 | 707.35 | 707.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 699.85 | 705.85 | 706.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 707.45 | 705.06 | 706.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 707.45 | 705.06 | 706.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 707.45 | 705.06 | 706.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 703.85 | 705.06 | 706.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 704.10 | 704.87 | 705.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 15:15:00 | 700.00 | 704.87 | 705.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:45:00 | 698.35 | 702.13 | 704.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 11:15:00 | 706.80 | 702.13 | 702.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 706.80 | 702.13 | 702.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 713.00 | 708.04 | 705.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 707.80 | 707.99 | 705.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 707.80 | 707.99 | 705.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 707.80 | 707.99 | 705.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 706.85 | 707.99 | 705.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 709.65 | 708.32 | 706.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 706.15 | 708.32 | 706.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 714.50 | 718.23 | 713.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 714.50 | 718.23 | 713.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 711.60 | 716.91 | 713.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 711.60 | 716.91 | 713.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 710.75 | 715.68 | 713.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:15:00 | 712.45 | 715.68 | 713.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:00:00 | 712.65 | 714.48 | 713.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 703.40 | 711.25 | 712.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 09:15:00 | 703.40 | 711.25 | 712.00 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 716.10 | 712.57 | 712.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 719.95 | 715.29 | 713.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 710.45 | 714.32 | 713.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 710.45 | 714.32 | 713.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 710.45 | 714.32 | 713.56 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 708.05 | 712.35 | 712.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 09:15:00 | 706.55 | 709.51 | 711.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 10:15:00 | 713.45 | 710.30 | 711.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 10:15:00 | 713.45 | 710.30 | 711.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 713.45 | 710.30 | 711.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 712.60 | 710.30 | 711.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 713.30 | 710.90 | 711.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:15:00 | 726.80 | 710.90 | 711.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 12:15:00 | 723.00 | 713.32 | 712.52 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 709.60 | 714.29 | 714.56 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 717.00 | 714.83 | 714.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 11:15:00 | 772.00 | 726.27 | 719.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 721.90 | 734.55 | 727.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 721.90 | 734.55 | 727.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 721.90 | 734.55 | 727.92 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 704.35 | 720.95 | 722.67 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 734.00 | 722.54 | 721.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 742.30 | 726.49 | 723.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 849.00 | 854.93 | 838.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:30:00 | 847.55 | 854.93 | 838.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 847.10 | 852.56 | 840.35 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 11:15:00 | 851.50 | 856.14 | 856.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 09:15:00 | 848.35 | 853.12 | 854.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 10:15:00 | 856.00 | 853.70 | 854.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 10:15:00 | 856.00 | 853.70 | 854.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 856.00 | 853.70 | 854.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 847.10 | 853.51 | 854.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 848.05 | 853.51 | 854.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:45:00 | 848.00 | 852.41 | 853.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 861.50 | 854.91 | 854.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 861.50 | 854.91 | 854.50 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 848.30 | 855.50 | 855.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 840.75 | 851.48 | 853.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 13:15:00 | 833.90 | 833.28 | 840.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 14:00:00 | 833.90 | 833.28 | 840.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 820.90 | 831.23 | 837.49 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 853.75 | 838.37 | 836.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 11:15:00 | 868.75 | 849.47 | 843.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 13:15:00 | 863.70 | 869.57 | 860.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 13:15:00 | 863.70 | 869.57 | 860.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 863.70 | 869.57 | 860.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:45:00 | 862.30 | 869.57 | 860.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 865.85 | 874.83 | 869.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 865.85 | 874.83 | 869.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 868.55 | 873.57 | 869.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:15:00 | 863.60 | 873.57 | 869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 871.40 | 873.14 | 869.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:30:00 | 870.45 | 873.14 | 869.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 865.35 | 871.58 | 869.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:45:00 | 865.05 | 871.58 | 869.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 862.10 | 869.68 | 868.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:45:00 | 862.70 | 869.68 | 868.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 860.25 | 866.66 | 867.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 837.00 | 860.72 | 864.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 867.15 | 845.56 | 852.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 867.15 | 845.56 | 852.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 867.15 | 845.56 | 852.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:00:00 | 857.00 | 851.40 | 853.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 862.00 | 855.43 | 855.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 862.00 | 855.43 | 855.28 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 849.75 | 855.16 | 855.24 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 864.70 | 856.54 | 855.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 867.40 | 858.71 | 856.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 10:15:00 | 871.20 | 871.60 | 865.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:45:00 | 869.50 | 871.60 | 865.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 860.00 | 872.18 | 868.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 860.00 | 872.18 | 868.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 855.30 | 868.81 | 867.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 855.30 | 868.81 | 867.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 853.50 | 865.75 | 866.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 13:15:00 | 849.05 | 860.38 | 863.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 856.25 | 855.43 | 859.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 11:00:00 | 856.25 | 855.43 | 859.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 860.90 | 856.52 | 859.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:00:00 | 860.90 | 856.52 | 859.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 859.20 | 857.06 | 859.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:45:00 | 861.50 | 857.06 | 859.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 866.05 | 858.86 | 860.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 866.05 | 858.86 | 860.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 862.90 | 859.67 | 860.59 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 15:15:00 | 868.00 | 861.33 | 861.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 907.95 | 870.66 | 865.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 11:15:00 | 925.10 | 925.22 | 911.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:30:00 | 925.35 | 925.22 | 911.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 964.90 | 935.35 | 919.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 964.90 | 935.35 | 919.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 954.00 | 968.94 | 954.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 946.75 | 968.94 | 954.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 956.35 | 966.42 | 955.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:15:00 | 962.00 | 966.42 | 955.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 961.40 | 965.42 | 955.58 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 942.85 | 951.28 | 951.85 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 969.90 | 955.50 | 953.54 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 950.50 | 963.23 | 964.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 11:15:00 | 946.00 | 959.78 | 962.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 951.35 | 943.61 | 952.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 951.35 | 943.61 | 952.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 951.35 | 943.61 | 952.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:00:00 | 918.90 | 935.16 | 943.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:45:00 | 919.15 | 932.09 | 940.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 14:45:00 | 921.10 | 929.67 | 939.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 872.95 | 905.97 | 919.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 873.19 | 905.97 | 919.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 875.04 | 905.97 | 919.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 862.35 | 862.26 | 878.71 | SL hit (close>ema200) qty=0.50 sl=862.26 alert=retest2 |

### Cycle 42 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 859.80 | 855.83 | 855.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 864.05 | 858.76 | 857.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 12:15:00 | 899.40 | 900.82 | 889.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 12:45:00 | 899.85 | 900.82 | 889.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 916.65 | 905.15 | 895.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:30:00 | 918.55 | 910.66 | 900.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 14:15:00 | 885.00 | 897.95 | 899.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 885.00 | 897.95 | 899.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 876.60 | 892.42 | 896.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 858.90 | 855.10 | 870.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 858.90 | 855.10 | 870.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 850.00 | 854.76 | 863.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 840.00 | 851.98 | 859.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 870.25 | 851.67 | 856.79 | SL hit (close>static) qty=1.00 sl=867.50 alert=retest2 |

### Cycle 44 — BUY (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 14:15:00 | 867.40 | 860.24 | 859.63 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 840.05 | 856.95 | 859.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 838.00 | 850.93 | 855.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 845.95 | 845.45 | 848.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 15:15:00 | 845.95 | 845.45 | 848.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 845.95 | 845.45 | 848.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 854.85 | 845.45 | 848.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 850.70 | 846.50 | 849.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 853.90 | 846.50 | 849.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 850.00 | 847.20 | 849.21 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 855.00 | 850.11 | 849.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 857.80 | 851.64 | 850.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 14:15:00 | 875.00 | 880.53 | 874.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 14:15:00 | 875.00 | 880.53 | 874.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 875.00 | 880.53 | 874.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 875.00 | 880.53 | 874.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 874.25 | 879.27 | 874.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 875.45 | 879.27 | 874.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 872.60 | 877.94 | 874.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 872.60 | 877.94 | 874.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 874.90 | 877.33 | 874.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 872.20 | 877.33 | 874.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 874.10 | 876.68 | 874.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 873.25 | 876.68 | 874.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 877.35 | 876.82 | 874.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:15:00 | 874.30 | 876.82 | 874.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 880.75 | 877.60 | 875.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 877.05 | 877.60 | 875.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 879.95 | 878.07 | 875.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 876.00 | 878.07 | 875.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 873.75 | 877.04 | 875.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 871.35 | 877.04 | 875.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 870.00 | 875.63 | 875.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:30:00 | 867.95 | 875.63 | 875.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 869.45 | 874.39 | 874.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 862.40 | 870.44 | 872.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 866.90 | 866.48 | 869.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 866.90 | 866.48 | 869.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 866.30 | 862.27 | 865.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:30:00 | 860.80 | 861.34 | 865.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 817.76 | 828.90 | 841.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-18 09:15:00 | 774.72 | 799.40 | 817.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 48 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 768.45 | 752.52 | 750.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 13:15:00 | 771.40 | 759.07 | 754.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 802.45 | 802.89 | 790.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 802.45 | 802.89 | 790.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 795.50 | 801.40 | 793.86 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 15:15:00 | 792.65 | 794.63 | 794.90 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 808.15 | 797.33 | 796.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 829.05 | 806.96 | 801.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 13:15:00 | 814.60 | 816.31 | 808.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 14:00:00 | 814.60 | 816.31 | 808.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 819.30 | 815.92 | 810.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 11:30:00 | 832.50 | 817.13 | 812.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:30:00 | 825.05 | 819.99 | 816.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 824.50 | 819.99 | 816.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 12:15:00 | 823.50 | 838.75 | 840.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 823.50 | 838.75 | 840.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 13:15:00 | 816.45 | 834.29 | 838.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 14:15:00 | 810.60 | 802.36 | 810.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 14:15:00 | 810.60 | 802.36 | 810.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 810.60 | 802.36 | 810.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 810.60 | 802.36 | 810.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 811.00 | 804.09 | 810.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 808.30 | 804.09 | 810.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 800.65 | 803.40 | 809.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 797.50 | 801.31 | 805.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 789.35 | 798.94 | 804.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:15:00 | 757.62 | 778.43 | 789.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 777.05 | 767.85 | 778.16 | SL hit (close>ema200) qty=0.50 sl=767.85 alert=retest2 |

### Cycle 52 — BUY (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 14:15:00 | 785.50 | 776.59 | 776.14 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 768.85 | 776.13 | 776.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 12:15:00 | 767.70 | 772.51 | 774.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 15:15:00 | 769.00 | 767.22 | 769.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 15:15:00 | 769.00 | 767.22 | 769.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 769.00 | 767.22 | 769.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 772.30 | 767.22 | 769.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 774.95 | 768.77 | 770.21 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 776.95 | 771.71 | 771.32 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 759.30 | 770.49 | 771.10 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 779.20 | 772.02 | 771.41 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 10:15:00 | 765.05 | 770.87 | 771.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 11:15:00 | 753.10 | 767.31 | 769.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 740.35 | 733.67 | 740.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 740.35 | 733.67 | 740.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 740.35 | 733.67 | 740.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 738.50 | 733.67 | 740.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 735.30 | 733.99 | 740.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:00:00 | 732.05 | 733.61 | 739.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 695.45 | 709.68 | 716.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 690.00 | 689.99 | 696.86 | SL hit (close>ema200) qty=0.50 sl=689.99 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 701.50 | 698.97 | 698.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 711.00 | 701.38 | 700.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 710.05 | 710.92 | 706.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 710.05 | 710.92 | 706.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 699.65 | 708.34 | 706.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 699.65 | 708.34 | 706.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 702.55 | 707.18 | 706.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 699.55 | 707.18 | 706.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 699.00 | 705.02 | 705.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 10:15:00 | 696.30 | 703.28 | 704.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 11:15:00 | 697.05 | 696.62 | 699.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 11:15:00 | 697.05 | 696.62 | 699.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 697.05 | 696.62 | 699.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:45:00 | 699.30 | 696.62 | 699.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 699.35 | 697.17 | 699.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:45:00 | 698.40 | 697.17 | 699.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 696.00 | 696.93 | 699.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 684.65 | 697.33 | 699.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:00:00 | 692.70 | 691.89 | 693.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 692.10 | 691.95 | 693.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 691.80 | 692.26 | 693.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 686.10 | 691.03 | 692.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:00:00 | 685.65 | 689.44 | 691.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 667.30 | 688.63 | 690.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 658.07 | 678.92 | 685.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 657.50 | 678.92 | 685.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 657.21 | 678.92 | 685.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 650.42 | 665.90 | 676.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 651.37 | 665.90 | 676.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 633.93 | 653.55 | 667.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 650.05 | 648.10 | 658.07 | SL hit (close>ema200) qty=0.50 sl=648.10 alert=retest2 |

### Cycle 60 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 685.95 | 663.19 | 660.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 690.35 | 681.31 | 673.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 725.10 | 728.51 | 716.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:15:00 | 720.95 | 728.51 | 716.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 719.95 | 723.34 | 718.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 15:15:00 | 722.00 | 723.34 | 718.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:00:00 | 723.20 | 724.19 | 720.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 10:15:00 | 722.25 | 725.10 | 722.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 657.00 | 716.72 | 721.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 657.00 | 716.72 | 721.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 637.50 | 647.61 | 668.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 603.60 | 603.20 | 617.78 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:15:00 | 600.50 | 603.20 | 617.78 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:45:00 | 600.95 | 602.85 | 616.30 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 11:30:00 | 600.70 | 602.43 | 614.88 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 12:00:00 | 600.75 | 602.43 | 614.88 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 593.00 | 598.13 | 607.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:30:00 | 616.35 | 598.13 | 607.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.48 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.90 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.66 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 570.71 | 582.92 | 594.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 583.00 | 582.94 | 593.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 583.00 | 582.94 | 593.20 | SL hit (close>ema200) qty=0.50 sl=582.94 alert=retest1 |

### Cycle 62 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 602.85 | 583.41 | 582.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 606.35 | 588.00 | 584.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 633.95 | 638.76 | 623.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 633.95 | 638.76 | 623.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 645.40 | 652.85 | 646.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 649.65 | 652.85 | 646.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 632.00 | 643.05 | 644.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 632.00 | 643.05 | 644.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 621.50 | 638.74 | 642.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 627.55 | 627.49 | 634.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 627.55 | 627.49 | 634.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 625.05 | 627.00 | 634.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 616.70 | 627.00 | 634.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:30:00 | 618.05 | 621.78 | 629.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 14:15:00 | 619.50 | 621.60 | 628.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 615.85 | 624.10 | 628.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 603.80 | 620.04 | 626.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 643.90 | 625.37 | 626.54 | SL hit (close>static) qty=1.00 sl=638.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 655.20 | 631.33 | 629.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 657.00 | 640.25 | 633.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 11:15:00 | 663.35 | 663.79 | 653.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 12:00:00 | 663.35 | 663.79 | 653.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 669.35 | 673.64 | 669.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 669.35 | 673.64 | 669.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 665.65 | 672.04 | 669.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 663.90 | 672.04 | 669.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 663.75 | 670.39 | 668.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 664.10 | 670.39 | 668.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 662.35 | 668.78 | 668.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:30:00 | 661.55 | 668.78 | 668.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 658.00 | 666.62 | 667.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 655.80 | 663.03 | 665.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 645.00 | 644.44 | 651.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 643.75 | 644.44 | 651.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 644.60 | 644.47 | 650.80 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 657.90 | 652.55 | 652.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 11:15:00 | 663.80 | 655.54 | 653.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 11:15:00 | 677.05 | 679.36 | 672.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 12:00:00 | 677.05 | 679.36 | 672.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 675.35 | 677.99 | 673.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:30:00 | 672.35 | 677.99 | 673.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 694.65 | 697.67 | 694.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 692.75 | 697.67 | 694.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 697.00 | 697.53 | 694.45 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 12:15:00 | 689.60 | 693.44 | 693.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 15:15:00 | 686.50 | 690.88 | 692.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 666.45 | 664.45 | 671.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:45:00 | 667.95 | 664.45 | 671.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 667.25 | 662.07 | 666.86 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 691.95 | 669.79 | 668.86 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 14:15:00 | 670.10 | 675.33 | 675.68 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 680.10 | 675.73 | 675.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 681.90 | 678.39 | 676.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 675.85 | 678.45 | 677.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 675.85 | 678.45 | 677.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 675.85 | 678.45 | 677.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:00:00 | 689.60 | 680.68 | 678.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 13:15:00 | 669.00 | 677.25 | 677.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 669.00 | 677.25 | 677.30 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 14:15:00 | 678.00 | 677.40 | 677.36 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 673.00 | 676.52 | 676.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 638.00 | 668.82 | 673.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 675.20 | 658.60 | 665.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 675.20 | 658.60 | 665.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 675.20 | 658.60 | 665.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 675.20 | 658.60 | 665.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 672.00 | 661.28 | 665.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 685.50 | 661.28 | 665.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 680.75 | 668.49 | 668.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 695.00 | 680.15 | 676.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 711.65 | 712.35 | 705.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 711.65 | 712.35 | 705.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 708.70 | 711.41 | 707.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:45:00 | 720.15 | 712.62 | 708.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:45:00 | 716.45 | 713.49 | 709.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 13:15:00 | 731.25 | 734.67 | 734.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 731.25 | 734.67 | 734.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 728.55 | 733.45 | 734.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 708.05 | 704.79 | 711.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 690.90 | 696.18 | 700.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 690.90 | 696.18 | 700.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 688.20 | 694.58 | 699.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 688.70 | 686.81 | 690.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 706.10 | 691.34 | 691.86 | SL hit (close>static) qty=1.00 sl=704.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 701.40 | 693.35 | 692.72 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 684.15 | 691.35 | 692.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 680.90 | 689.26 | 691.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 701.25 | 681.13 | 683.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 701.25 | 681.13 | 683.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 701.25 | 681.13 | 683.75 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 701.00 | 688.38 | 686.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 704.15 | 691.53 | 688.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 692.15 | 699.33 | 695.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 692.15 | 699.33 | 695.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 692.15 | 699.33 | 695.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 692.15 | 699.33 | 695.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 693.05 | 698.07 | 694.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 690.35 | 698.07 | 694.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 690.10 | 696.48 | 694.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 690.10 | 696.48 | 694.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 704.85 | 709.22 | 705.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 14:45:00 | 710.60 | 707.89 | 705.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 13:15:00 | 711.50 | 716.10 | 716.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 711.50 | 716.10 | 716.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 709.60 | 713.90 | 715.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 718.00 | 713.73 | 714.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 718.00 | 713.73 | 714.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 718.00 | 713.73 | 714.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 721.65 | 713.73 | 714.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 724.55 | 715.90 | 715.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 734.00 | 729.94 | 726.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 757.55 | 759.89 | 750.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 757.55 | 759.89 | 750.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 762.45 | 764.99 | 758.29 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 753.85 | 759.30 | 759.75 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 763.50 | 760.12 | 759.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 765.00 | 761.75 | 760.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 753.90 | 761.50 | 760.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 753.90 | 761.50 | 760.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 753.90 | 761.50 | 760.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 753.90 | 761.50 | 760.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 756.85 | 760.57 | 760.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 760.35 | 760.57 | 760.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 760.00 | 760.66 | 760.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 760.00 | 760.66 | 760.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 760.70 | 760.67 | 760.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:00:00 | 760.70 | 760.67 | 760.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 759.65 | 760.47 | 760.54 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 763.75 | 761.12 | 760.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 14:15:00 | 765.70 | 762.04 | 761.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 759.15 | 761.97 | 761.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 759.15 | 761.97 | 761.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 759.15 | 761.97 | 761.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 755.20 | 761.97 | 761.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 758.00 | 761.17 | 761.10 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 758.15 | 760.57 | 760.83 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 763.00 | 760.62 | 760.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 768.25 | 762.53 | 761.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 758.50 | 761.95 | 761.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 11:15:00 | 758.50 | 761.95 | 761.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 758.50 | 761.95 | 761.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 757.10 | 761.95 | 761.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 753.05 | 760.17 | 760.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 746.65 | 757.46 | 759.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 741.80 | 741.62 | 745.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 741.80 | 741.62 | 745.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 741.80 | 741.62 | 745.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 737.65 | 740.74 | 744.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 700.77 | 712.86 | 722.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 711.75 | 711.35 | 719.08 | SL hit (close>ema200) qty=0.50 sl=711.35 alert=retest2 |

### Cycle 88 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 715.80 | 711.50 | 711.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 732.30 | 718.85 | 715.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 747.30 | 748.01 | 738.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 739.60 | 744.67 | 738.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 739.60 | 744.67 | 738.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 742.00 | 740.81 | 738.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:45:00 | 741.90 | 740.28 | 739.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 742.60 | 740.47 | 739.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 741.25 | 740.47 | 739.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 738.40 | 740.43 | 739.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 734.55 | 740.43 | 739.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 744.20 | 741.18 | 739.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-03 13:15:00 | 738.55 | 740.33 | 740.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 738.55 | 740.33 | 740.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 15:15:00 | 736.05 | 739.16 | 739.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 735.25 | 734.02 | 736.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 735.25 | 734.02 | 736.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 735.25 | 734.02 | 736.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 735.25 | 734.02 | 736.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 727.95 | 732.96 | 735.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 731.10 | 732.96 | 735.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 734.70 | 731.64 | 733.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 735.55 | 731.64 | 733.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 729.90 | 731.30 | 733.02 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 737.40 | 733.86 | 733.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 741.00 | 735.87 | 734.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 735.20 | 736.92 | 735.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 735.20 | 736.92 | 735.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 735.20 | 736.92 | 735.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 735.20 | 736.92 | 735.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 735.40 | 736.61 | 735.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 736.05 | 736.61 | 735.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 732.00 | 735.37 | 735.15 | SL hit (close<static) qty=1.00 sl=733.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 731.00 | 734.50 | 734.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 728.95 | 732.33 | 733.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 724.35 | 722.79 | 725.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 724.35 | 722.79 | 725.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 724.35 | 722.79 | 725.93 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 741.10 | 728.68 | 727.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 752.70 | 747.94 | 742.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 753.40 | 754.96 | 749.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 750.10 | 754.96 | 749.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 749.30 | 753.82 | 749.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 747.90 | 753.82 | 749.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 750.00 | 753.06 | 749.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 754.35 | 753.06 | 749.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 753.00 | 752.05 | 749.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 755.85 | 771.63 | 773.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 755.85 | 771.63 | 773.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 751.25 | 767.55 | 771.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 762.75 | 757.23 | 763.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 762.75 | 757.23 | 763.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 762.75 | 757.23 | 763.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 764.90 | 757.23 | 763.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 757.45 | 757.28 | 762.62 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 766.20 | 764.34 | 764.29 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 760.00 | 763.97 | 764.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 751.90 | 761.62 | 763.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 754.25 | 753.67 | 757.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 754.25 | 753.67 | 757.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 754.25 | 753.67 | 757.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 757.95 | 753.67 | 757.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 757.20 | 754.37 | 757.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 753.05 | 755.85 | 757.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 715.40 | 722.92 | 727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 722.50 | 722.02 | 726.09 | SL hit (close>ema200) qty=0.50 sl=722.02 alert=retest2 |

### Cycle 96 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 760.00 | 728.81 | 724.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 769.90 | 737.03 | 728.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 756.60 | 759.12 | 745.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:30:00 | 757.25 | 759.12 | 745.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 778.00 | 784.04 | 777.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 778.00 | 784.04 | 777.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 779.00 | 783.03 | 778.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 769.00 | 783.03 | 778.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 772.85 | 780.99 | 777.57 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 771.00 | 775.62 | 775.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 770.00 | 774.49 | 775.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 775.75 | 773.40 | 774.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 775.75 | 773.40 | 774.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 775.75 | 773.40 | 774.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 775.75 | 773.40 | 774.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 789.70 | 776.66 | 775.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 802.45 | 783.68 | 779.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 792.00 | 792.77 | 788.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 774.20 | 792.77 | 788.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 782.05 | 790.62 | 787.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:30:00 | 789.45 | 791.59 | 788.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 780.30 | 788.36 | 788.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 780.30 | 788.36 | 788.79 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 799.35 | 790.23 | 789.25 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 787.70 | 792.98 | 792.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 14:15:00 | 785.55 | 789.24 | 790.93 | Break + close below crossover candle low |

### Cycle 102 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 804.30 | 792.20 | 791.98 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 784.50 | 795.20 | 796.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 769.65 | 783.98 | 788.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 775.65 | 775.35 | 779.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 775.65 | 775.35 | 779.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 772.05 | 774.69 | 777.42 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 788.95 | 779.99 | 778.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 790.00 | 781.99 | 779.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 789.05 | 794.06 | 790.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 789.05 | 794.06 | 790.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 789.05 | 794.06 | 790.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 789.05 | 794.06 | 790.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 787.70 | 792.79 | 789.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:15:00 | 786.60 | 792.79 | 789.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 789.00 | 790.21 | 789.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 789.00 | 790.21 | 789.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 788.55 | 789.88 | 789.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 785.00 | 789.88 | 789.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 782.50 | 788.40 | 788.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 777.35 | 783.51 | 786.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 788.80 | 781.87 | 783.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 788.80 | 781.87 | 783.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 788.80 | 781.87 | 783.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 788.80 | 781.87 | 783.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 787.90 | 783.08 | 783.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 782.00 | 783.08 | 783.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:30:00 | 786.15 | 783.75 | 783.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 742.90 | 752.35 | 760.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 746.84 | 752.35 | 760.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 731.50 | 728.21 | 734.68 | SL hit (close>ema200) qty=0.50 sl=728.21 alert=retest2 |

### Cycle 106 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 733.20 | 730.01 | 729.67 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 724.50 | 729.57 | 729.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 722.30 | 728.12 | 729.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 729.50 | 726.68 | 727.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 729.50 | 726.68 | 727.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 729.50 | 726.68 | 727.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 729.50 | 726.68 | 727.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 733.55 | 728.05 | 728.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 733.55 | 728.05 | 728.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 732.30 | 728.90 | 728.75 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 725.95 | 728.87 | 729.16 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 12:15:00 | 730.35 | 728.30 | 728.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 13:15:00 | 737.75 | 730.19 | 729.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 736.80 | 739.04 | 733.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 736.80 | 739.04 | 733.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 734.85 | 738.20 | 734.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 734.85 | 738.20 | 734.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 734.80 | 737.52 | 734.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 734.80 | 737.52 | 734.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 731.25 | 736.27 | 733.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 731.25 | 736.27 | 733.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 733.30 | 735.67 | 733.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 737.00 | 734.70 | 733.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 735.00 | 735.47 | 734.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 737.05 | 735.26 | 734.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 735.95 | 735.00 | 734.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 734.95 | 734.99 | 734.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 733.30 | 734.28 | 734.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 733.30 | 734.28 | 734.38 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 736.95 | 734.41 | 734.28 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 730.00 | 733.53 | 733.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 727.00 | 730.71 | 732.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 10:15:00 | 729.55 | 728.74 | 730.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 10:15:00 | 729.55 | 728.74 | 730.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 729.55 | 728.74 | 730.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:15:00 | 726.30 | 728.98 | 729.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 726.70 | 728.51 | 729.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 734.60 | 730.33 | 730.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 734.60 | 730.33 | 730.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 736.45 | 732.44 | 731.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 731.20 | 732.89 | 732.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 731.20 | 732.89 | 732.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 731.20 | 732.89 | 732.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 732.10 | 732.89 | 732.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 727.65 | 731.84 | 731.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 727.65 | 731.84 | 731.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 729.80 | 731.43 | 731.51 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 733.80 | 731.91 | 731.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 735.00 | 732.52 | 732.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 731.25 | 732.27 | 731.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 731.25 | 732.27 | 731.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 731.25 | 732.27 | 731.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 731.25 | 732.27 | 731.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 730.00 | 731.82 | 731.77 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 730.00 | 731.45 | 731.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 728.25 | 730.54 | 731.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 11:15:00 | 732.50 | 728.87 | 729.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 11:15:00 | 732.50 | 728.87 | 729.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 732.50 | 728.87 | 729.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 732.50 | 728.87 | 729.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 731.25 | 729.34 | 729.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 731.60 | 729.34 | 729.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 729.25 | 729.27 | 729.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 731.15 | 729.27 | 729.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 732.30 | 729.88 | 729.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 727.35 | 729.53 | 729.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 726.70 | 729.53 | 729.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:00:00 | 726.55 | 714.20 | 715.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 727.10 | 718.93 | 717.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 727.10 | 718.93 | 717.93 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 712.25 | 717.06 | 717.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 10:15:00 | 709.80 | 714.63 | 716.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 709.80 | 709.10 | 711.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 12:00:00 | 709.80 | 709.10 | 711.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 715.65 | 710.41 | 712.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 715.65 | 710.41 | 712.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 713.70 | 711.07 | 712.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 710.75 | 711.32 | 712.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 712.60 | 711.92 | 712.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 14:15:00 | 714.40 | 712.48 | 712.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 714.40 | 712.48 | 712.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 715.90 | 713.16 | 712.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 712.75 | 718.92 | 717.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 712.75 | 718.92 | 717.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 712.75 | 718.92 | 717.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 713.20 | 718.92 | 717.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 717.35 | 718.61 | 717.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 712.85 | 718.61 | 717.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 718.05 | 718.50 | 717.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:45:00 | 720.00 | 718.81 | 717.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 720.20 | 718.87 | 717.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 713.65 | 719.90 | 719.69 | SL hit (close<static) qty=1.00 sl=715.40 alert=retest2 |

### Cycle 121 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 716.30 | 719.18 | 719.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 711.90 | 716.38 | 717.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 708.95 | 708.10 | 710.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 708.95 | 708.10 | 710.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 708.95 | 708.10 | 710.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 709.70 | 708.10 | 710.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 709.50 | 707.69 | 709.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 709.50 | 707.69 | 709.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 707.65 | 707.68 | 709.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 708.85 | 707.68 | 709.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 712.30 | 708.61 | 709.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 712.30 | 708.61 | 709.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 712.45 | 709.38 | 710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 712.45 | 709.38 | 710.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 710.60 | 710.15 | 710.29 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 713.75 | 710.99 | 710.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 718.50 | 713.51 | 712.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 708.70 | 713.27 | 712.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 708.70 | 713.27 | 712.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 708.70 | 713.27 | 712.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 708.70 | 713.27 | 712.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 709.50 | 712.51 | 712.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 708.35 | 712.51 | 712.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 711.40 | 712.15 | 712.18 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 715.50 | 712.82 | 712.48 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 705.40 | 712.15 | 712.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 701.40 | 710.00 | 711.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 701.05 | 699.23 | 703.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 701.05 | 699.23 | 703.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 702.80 | 699.80 | 702.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 702.05 | 699.80 | 702.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 711.75 | 702.19 | 703.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 711.75 | 702.19 | 703.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 712.00 | 704.15 | 704.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 12:15:00 | 716.95 | 706.71 | 705.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 711.55 | 714.21 | 711.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 711.55 | 714.21 | 711.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 711.55 | 714.21 | 711.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 711.55 | 714.21 | 711.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 710.15 | 713.40 | 711.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 705.65 | 713.40 | 711.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 710.10 | 712.74 | 711.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 710.10 | 712.74 | 711.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 705.00 | 709.51 | 710.03 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 724.80 | 708.81 | 708.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 730.30 | 719.66 | 714.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 715.25 | 720.03 | 715.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 715.25 | 720.03 | 715.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 715.25 | 720.03 | 715.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 729.40 | 722.79 | 719.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:30:00 | 729.30 | 725.04 | 721.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 15:15:00 | 730.65 | 725.70 | 721.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 09:30:00 | 730.80 | 727.38 | 723.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 744.80 | 738.80 | 732.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 746.90 | 738.80 | 732.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 745.05 | 743.67 | 737.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 746.00 | 745.11 | 739.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:45:00 | 745.55 | 742.84 | 740.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 741.65 | 742.73 | 741.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 741.65 | 742.73 | 741.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 743.75 | 742.93 | 741.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 744.70 | 743.90 | 742.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:45:00 | 748.80 | 747.02 | 744.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 742.60 | 746.41 | 746.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 742.60 | 746.41 | 746.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 740.50 | 745.15 | 745.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 745.30 | 744.07 | 745.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 745.30 | 744.07 | 745.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 745.30 | 744.07 | 745.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 746.10 | 744.07 | 745.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 745.40 | 744.34 | 745.20 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 750.15 | 745.96 | 745.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 09:15:00 | 754.85 | 749.34 | 747.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 748.25 | 749.12 | 747.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 748.25 | 749.12 | 747.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 748.25 | 749.12 | 747.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 747.70 | 749.12 | 747.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 756.05 | 750.50 | 748.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:30:00 | 757.85 | 753.63 | 750.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 757.95 | 755.73 | 752.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 749.40 | 754.59 | 754.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 749.40 | 754.59 | 754.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 09:15:00 | 749.00 | 751.55 | 752.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 716.55 | 715.35 | 721.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 716.55 | 715.35 | 721.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 716.55 | 715.35 | 721.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 722.00 | 715.35 | 721.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 721.80 | 716.64 | 721.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 721.80 | 716.64 | 721.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 725.20 | 718.35 | 722.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 725.20 | 718.35 | 722.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 723.50 | 719.38 | 722.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:45:00 | 723.00 | 719.38 | 722.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 722.90 | 720.08 | 722.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 716.65 | 719.58 | 721.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-20 09:15:00 | 644.99 | 693.99 | 696.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 668.00 | 661.21 | 660.64 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 655.30 | 659.44 | 659.90 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 664.15 | 660.83 | 660.42 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 658.00 | 660.31 | 660.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 654.40 | 659.13 | 659.78 | Break + close below crossover candle low |

### Cycle 136 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 668.80 | 661.07 | 660.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 673.10 | 663.47 | 661.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 660.50 | 664.54 | 663.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 660.50 | 664.54 | 663.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 660.50 | 664.54 | 663.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 660.50 | 664.54 | 663.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 660.50 | 663.73 | 662.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 647.15 | 663.73 | 662.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 653.40 | 661.66 | 661.93 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 664.00 | 659.10 | 659.02 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 12:15:00 | 657.80 | 659.93 | 659.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 652.05 | 657.56 | 658.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 15:15:00 | 657.60 | 655.83 | 657.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 15:15:00 | 657.60 | 655.83 | 657.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 657.60 | 655.83 | 657.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 647.95 | 655.83 | 657.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 660.35 | 652.23 | 653.06 | SL hit (close>static) qty=1.00 sl=658.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 661.95 | 654.17 | 653.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 665.60 | 656.46 | 654.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 667.40 | 670.19 | 665.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 667.40 | 670.19 | 665.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 667.40 | 670.19 | 665.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 664.05 | 670.19 | 665.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 672.25 | 670.60 | 665.98 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 659.15 | 664.36 | 664.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 654.60 | 661.07 | 663.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 640.55 | 636.99 | 643.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 640.55 | 636.99 | 643.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 642.00 | 637.97 | 641.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:00:00 | 642.00 | 637.97 | 641.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 641.75 | 638.72 | 641.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 641.75 | 638.72 | 641.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 639.90 | 638.96 | 641.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:15:00 | 644.00 | 638.96 | 641.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 644.00 | 639.97 | 641.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:30:00 | 638.00 | 640.69 | 641.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 638.45 | 639.76 | 641.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 637.00 | 639.81 | 641.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 638.70 | 638.95 | 640.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 640.50 | 639.26 | 640.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 640.50 | 639.26 | 640.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 641.35 | 639.68 | 640.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:45:00 | 641.85 | 639.68 | 640.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 639.80 | 639.70 | 640.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 639.80 | 639.70 | 640.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 641.50 | 640.06 | 640.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 637.45 | 640.06 | 640.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 633.00 | 629.86 | 629.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 633.00 | 629.86 | 629.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 10:15:00 | 637.35 | 631.36 | 630.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 630.75 | 634.58 | 632.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 630.75 | 634.58 | 632.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 630.75 | 634.58 | 632.82 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 629.95 | 631.66 | 631.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 624.70 | 630.09 | 631.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 629.95 | 628.80 | 630.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:00:00 | 629.95 | 628.80 | 630.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 630.00 | 629.04 | 630.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 630.00 | 629.04 | 630.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 630.80 | 629.39 | 630.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 630.65 | 629.39 | 630.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 630.00 | 629.51 | 630.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 632.00 | 629.51 | 630.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 633.05 | 630.22 | 630.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 632.00 | 630.22 | 630.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 632.55 | 630.69 | 630.60 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 14:15:00 | 630.20 | 630.60 | 630.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 09:15:00 | 618.40 | 628.06 | 629.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 15:15:00 | 608.05 | 607.03 | 613.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 09:15:00 | 613.90 | 607.03 | 613.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 615.00 | 608.62 | 613.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 616.00 | 608.62 | 613.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 614.75 | 609.85 | 613.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 616.50 | 609.85 | 613.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 616.90 | 614.38 | 614.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 616.95 | 614.38 | 614.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 615.35 | 614.57 | 614.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 613.25 | 614.57 | 614.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 618.95 | 615.45 | 615.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 618.95 | 615.45 | 615.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 633.10 | 618.98 | 616.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 620.20 | 621.36 | 618.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 620.20 | 621.36 | 618.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 615.30 | 620.15 | 618.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 612.05 | 620.15 | 618.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 612.90 | 618.70 | 617.85 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 611.00 | 617.08 | 617.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 610.25 | 614.88 | 616.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 609.45 | 604.62 | 608.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 12:15:00 | 609.45 | 604.62 | 608.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 609.45 | 604.62 | 608.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 609.45 | 604.62 | 608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 618.00 | 607.30 | 609.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 618.00 | 607.30 | 609.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 623.00 | 610.44 | 610.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 623.00 | 610.44 | 610.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 15:15:00 | 623.50 | 613.05 | 611.94 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 611.05 | 614.94 | 615.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 606.50 | 609.92 | 612.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 610.85 | 609.29 | 611.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 610.85 | 609.29 | 611.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 610.85 | 609.29 | 611.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 611.40 | 609.29 | 611.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 622.00 | 611.83 | 612.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 622.00 | 611.83 | 612.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 632.00 | 615.86 | 613.93 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 607.40 | 613.86 | 613.97 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 620.15 | 613.08 | 612.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 626.40 | 615.89 | 614.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 620.35 | 622.75 | 619.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 620.35 | 622.75 | 619.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 620.35 | 622.75 | 619.38 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 598.40 | 615.15 | 617.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 596.35 | 611.39 | 615.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 623.75 | 610.25 | 612.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 623.75 | 610.25 | 612.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 623.75 | 610.25 | 612.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 623.75 | 610.25 | 612.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 628.00 | 613.80 | 614.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 626.25 | 613.80 | 614.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 623.95 | 615.83 | 615.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 635.15 | 624.62 | 622.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 628.60 | 628.81 | 626.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:30:00 | 628.15 | 628.81 | 626.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 626.95 | 628.44 | 626.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 634.15 | 631.39 | 627.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 10:15:00 | 630.80 | 637.66 | 638.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 630.80 | 637.66 | 638.35 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 644.25 | 639.10 | 638.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 647.20 | 642.01 | 640.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 657.20 | 660.50 | 654.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 657.20 | 660.50 | 654.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 654.95 | 659.39 | 654.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 654.95 | 659.39 | 654.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 651.80 | 657.87 | 654.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 656.80 | 657.66 | 654.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 660.70 | 657.66 | 654.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 13:45:00 | 657.85 | 657.34 | 655.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 663.15 | 677.82 | 678.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 663.15 | 677.82 | 678.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 661.65 | 674.59 | 676.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 672.05 | 669.93 | 673.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 672.05 | 669.93 | 673.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 672.05 | 669.93 | 673.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 673.10 | 669.93 | 673.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 672.50 | 670.44 | 673.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 672.50 | 670.44 | 673.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 671.40 | 670.63 | 673.24 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 680.70 | 675.11 | 674.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 681.65 | 677.20 | 675.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 672.40 | 677.82 | 676.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 14:15:00 | 672.40 | 677.82 | 676.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 672.40 | 677.82 | 676.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 672.40 | 677.82 | 676.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 678.60 | 677.98 | 677.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 681.85 | 677.98 | 677.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 682.80 | 679.63 | 677.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 675.25 | 680.09 | 680.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 675.25 | 680.09 | 680.58 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 684.40 | 680.13 | 680.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 687.30 | 682.41 | 681.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 680.75 | 683.30 | 682.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 680.75 | 683.30 | 682.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 680.75 | 683.30 | 682.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 680.75 | 683.30 | 682.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 677.55 | 682.15 | 681.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 676.20 | 682.15 | 681.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 680.10 | 681.55 | 681.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 680.00 | 681.55 | 681.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 680.00 | 681.24 | 681.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 677.00 | 680.24 | 680.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 12:15:00 | 679.95 | 679.91 | 680.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 12:15:00 | 679.95 | 679.91 | 680.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 679.95 | 679.91 | 680.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:45:00 | 679.30 | 679.91 | 680.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 681.10 | 680.15 | 680.58 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 682.70 | 681.15 | 680.99 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 677.30 | 680.38 | 680.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 10:15:00 | 675.45 | 679.40 | 680.18 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 10:15:00 | 575.95 | 2024-06-04 11:15:00 | 551.05 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2024-06-20 09:15:00 | 732.20 | 2024-06-20 13:15:00 | 718.70 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-06-24 11:15:00 | 722.80 | 2024-06-24 11:15:00 | 721.05 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-06-25 09:15:00 | 721.45 | 2024-06-26 13:15:00 | 728.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-06-25 11:15:00 | 720.35 | 2024-06-26 13:15:00 | 728.95 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-06-26 12:00:00 | 721.55 | 2024-06-26 13:15:00 | 728.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-07-05 10:30:00 | 714.70 | 2024-07-11 11:15:00 | 702.80 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2024-07-05 11:00:00 | 714.85 | 2024-07-11 11:15:00 | 702.80 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-07-18 09:15:00 | 718.55 | 2024-07-18 09:15:00 | 703.25 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-07-18 15:15:00 | 700.00 | 2024-07-22 11:15:00 | 706.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-07-19 09:45:00 | 698.35 | 2024-07-22 11:15:00 | 706.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-07-25 12:15:00 | 712.45 | 2024-07-26 09:15:00 | 703.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-07-25 14:00:00 | 712.65 | 2024-07-26 09:15:00 | 703.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-08-23 09:45:00 | 847.10 | 2024-08-26 09:15:00 | 861.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-08-23 10:15:00 | 848.05 | 2024-08-26 09:15:00 | 861.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-08-23 10:45:00 | 848.00 | 2024-08-26 09:15:00 | 861.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-09-10 12:00:00 | 857.00 | 2024-09-10 14:15:00 | 862.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-10-03 13:00:00 | 918.90 | 2024-10-07 09:15:00 | 872.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 13:45:00 | 919.15 | 2024-10-07 09:15:00 | 873.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 14:45:00 | 921.10 | 2024-10-07 09:15:00 | 875.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 13:00:00 | 918.90 | 2024-10-08 14:15:00 | 862.35 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2024-10-03 13:45:00 | 919.15 | 2024-10-08 14:15:00 | 862.35 | STOP_HIT | 0.50 | 6.18% |
| SELL | retest2 | 2024-10-03 14:45:00 | 921.10 | 2024-10-08 14:15:00 | 862.35 | STOP_HIT | 0.50 | 6.38% |
| BUY | retest2 | 2024-10-18 12:30:00 | 918.55 | 2024-10-21 14:15:00 | 885.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2024-10-24 14:15:00 | 840.00 | 2024-10-25 09:15:00 | 870.25 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-11-12 10:30:00 | 860.80 | 2024-11-14 09:15:00 | 817.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:30:00 | 860.80 | 2024-11-18 09:15:00 | 774.72 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-06 11:30:00 | 832.50 | 2024-12-12 12:15:00 | 823.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-12-09 09:30:00 | 825.05 | 2024-12-12 12:15:00 | 823.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-12-09 10:15:00 | 824.50 | 2024-12-12 12:15:00 | 823.50 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-12-18 09:15:00 | 797.50 | 2024-12-19 10:15:00 | 757.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:15:00 | 797.50 | 2024-12-20 09:15:00 | 777.05 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest2 | 2024-12-18 09:45:00 | 789.35 | 2024-12-23 14:15:00 | 785.50 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-01-07 12:00:00 | 732.05 | 2025-01-13 09:15:00 | 695.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 12:00:00 | 732.05 | 2025-01-14 15:15:00 | 690.00 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-01-22 09:15:00 | 684.65 | 2025-01-27 10:15:00 | 658.07 | PARTIAL | 0.50 | 3.88% |
| SELL | retest2 | 2025-01-23 13:00:00 | 692.70 | 2025-01-27 10:15:00 | 657.50 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-01-23 15:00:00 | 692.10 | 2025-01-27 10:15:00 | 657.21 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-01-24 09:15:00 | 691.80 | 2025-01-27 14:15:00 | 650.42 | PARTIAL | 0.50 | 5.98% |
| SELL | retest2 | 2025-01-24 12:00:00 | 685.65 | 2025-01-27 14:15:00 | 651.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 667.30 | 2025-01-28 10:15:00 | 633.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 684.65 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2025-01-23 13:00:00 | 692.70 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 6.16% |
| SELL | retest2 | 2025-01-23 15:00:00 | 692.10 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 6.08% |
| SELL | retest2 | 2025-01-24 09:15:00 | 691.80 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2025-01-24 12:00:00 | 685.65 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 5.19% |
| SELL | retest2 | 2025-01-27 09:15:00 | 667.30 | 2025-01-29 09:15:00 | 650.05 | STOP_HIT | 0.50 | 2.59% |
| BUY | retest2 | 2025-02-04 15:15:00 | 722.00 | 2025-02-07 09:15:00 | 657.00 | STOP_HIT | 1.00 | -9.00% |
| BUY | retest2 | 2025-02-05 12:00:00 | 723.20 | 2025-02-07 09:15:00 | 657.00 | STOP_HIT | 1.00 | -9.15% |
| BUY | retest2 | 2025-02-06 10:15:00 | 722.25 | 2025-02-07 09:15:00 | 657.00 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest1 | 2025-02-13 10:15:00 | 600.50 | 2025-02-17 09:15:00 | 570.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 10:45:00 | 600.95 | 2025-02-17 09:15:00 | 570.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 11:30:00 | 600.70 | 2025-02-17 09:15:00 | 570.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 12:00:00 | 600.75 | 2025-02-17 09:15:00 | 570.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 10:15:00 | 600.50 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest1 | 2025-02-13 10:45:00 | 600.95 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest1 | 2025-02-13 11:30:00 | 600.70 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest1 | 2025-02-13 12:00:00 | 600.75 | 2025-02-17 10:15:00 | 583.00 | STOP_HIT | 0.50 | 2.95% |
| BUY | retest2 | 2025-02-25 09:15:00 | 649.65 | 2025-02-27 09:15:00 | 632.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-02-28 09:15:00 | 616.70 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2025-02-28 12:30:00 | 618.05 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-02-28 14:15:00 | 619.50 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-03-03 09:15:00 | 615.85 | 2025-03-03 13:15:00 | 643.90 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-04-04 11:00:00 | 689.60 | 2025-04-04 13:15:00 | 669.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-04-17 11:45:00 | 720.15 | 2025-04-25 13:15:00 | 731.25 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-04-17 12:45:00 | 716.45 | 2025-04-25 13:15:00 | 731.25 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2025-05-06 11:00:00 | 688.20 | 2025-05-08 09:15:00 | 706.10 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-05-07 14:30:00 | 688.70 | 2025-05-08 09:15:00 | 706.10 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-05-16 14:45:00 | 710.60 | 2025-05-21 13:15:00 | 711.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-06-17 10:45:00 | 737.65 | 2025-06-19 13:15:00 | 700.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 10:45:00 | 737.65 | 2025-06-20 09:15:00 | 711.75 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-07-01 10:15:00 | 742.00 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-07-01 13:45:00 | 741.90 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-01 14:30:00 | 742.60 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-07-01 15:00:00 | 741.25 | 2025-07-03 13:15:00 | 738.55 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-07-10 09:15:00 | 736.05 | 2025-07-10 10:15:00 | 732.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-07-21 11:15:00 | 754.35 | 2025-07-28 09:15:00 | 755.85 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-07-21 15:15:00 | 753.00 | 2025-07-28 09:15:00 | 755.85 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-08-01 15:15:00 | 753.05 | 2025-08-11 09:15:00 | 715.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-01 15:15:00 | 753.05 | 2025-08-11 11:15:00 | 722.50 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest2 | 2025-08-28 10:30:00 | 789.45 | 2025-08-29 10:15:00 | 780.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-22 09:15:00 | 782.00 | 2025-09-25 14:15:00 | 742.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:30:00 | 786.15 | 2025-09-25 14:15:00 | 746.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 782.00 | 2025-09-30 10:15:00 | 731.50 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2025-09-22 11:30:00 | 786.15 | 2025-09-30 10:15:00 | 731.50 | STOP_HIT | 0.50 | 6.95% |
| BUY | retest2 | 2025-10-16 09:15:00 | 737.00 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-16 14:30:00 | 735.00 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-10-17 09:15:00 | 737.05 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-17 10:15:00 | 735.95 | 2025-10-17 13:15:00 | 733.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-28 10:15:00 | 726.30 | 2025-10-28 13:15:00 | 734.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-28 12:00:00 | 726.70 | 2025-10-28 13:15:00 | 734.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-11-06 09:30:00 | 727.35 | 2025-11-11 13:15:00 | 727.10 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-11-06 10:15:00 | 726.70 | 2025-11-11 13:15:00 | 727.10 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-11 12:00:00 | 726.55 | 2025-11-11 13:15:00 | 727.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-11-14 14:45:00 | 710.75 | 2025-11-17 14:15:00 | 714.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-17 13:15:00 | 712.60 | 2025-11-17 14:15:00 | 714.40 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-19 12:45:00 | 720.00 | 2025-11-21 10:15:00 | 713.65 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-19 15:00:00 | 720.20 | 2025-11-21 10:15:00 | 713.65 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-12 10:45:00 | 729.40 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2025-12-12 13:30:00 | 729.30 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2025-12-12 15:15:00 | 730.65 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-12-15 09:30:00 | 730.80 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-12-16 10:15:00 | 746.90 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-16 14:00:00 | 745.05 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-12-17 09:45:00 | 746.00 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-18 10:45:00 | 745.55 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-12-19 09:30:00 | 744.70 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-12-22 09:45:00 | 748.80 | 2025-12-24 09:15:00 | 742.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-29 13:30:00 | 757.85 | 2025-12-31 12:15:00 | 749.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-30 10:00:00 | 757.95 | 2025-12-31 12:15:00 | 749.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-08 10:45:00 | 716.65 | 2026-01-20 09:15:00 | 644.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-06 09:15:00 | 647.95 | 2026-02-09 10:15:00 | 660.35 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-02-18 11:30:00 | 638.00 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2026-02-18 14:45:00 | 638.45 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2026-02-19 09:15:00 | 637.00 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2026-02-19 12:15:00 | 638.70 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2026-02-20 09:15:00 | 637.45 | 2026-02-27 09:15:00 | 633.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-03-11 10:15:00 | 613.25 | 2026-03-11 10:15:00 | 618.95 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-04-08 09:30:00 | 634.15 | 2026-04-13 10:15:00 | 630.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-04-21 10:00:00 | 656.80 | 2026-04-24 12:15:00 | 663.15 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2026-04-21 10:30:00 | 660.70 | 2026-04-24 12:15:00 | 663.15 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2026-04-21 13:45:00 | 657.85 | 2026-04-24 12:15:00 | 663.15 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2026-04-29 09:15:00 | 681.85 | 2026-04-30 12:15:00 | 675.25 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-04-29 09:45:00 | 682.80 | 2026-04-30 12:15:00 | 675.25 | STOP_HIT | 1.00 | -1.11% |
