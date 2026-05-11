# Sun TV Network Ltd. (SUNTV)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 572.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 151 |
| ALERT1 | 104 |
| ALERT2 | 101 |
| ALERT2_SKIP | 44 |
| ALERT3 | 287 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 116 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 117 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 85
- **Target hits / Stop hits / Partials:** 4 / 116 / 11
- **Avg / median % per leg:** 0.21% / -0.78%
- **Sum % (uncompounded):** 27.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 8 | 17.8% | 0 | 44 | 1 | -0.73% | -32.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.88% | 7.5% |
| BUY @ 3rd Alert (retest2) | 41 | 6 | 14.6% | 0 | 41 | 0 | -0.98% | -40.4% |
| SELL (all) | 86 | 38 | 44.2% | 4 | 72 | 10 | 0.70% | 60.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.20% | -2.2% |
| SELL @ 3rd Alert (retest2) | 85 | 38 | 44.7% | 4 | 71 | 10 | 0.74% | 62.8% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 1.07% | 5.3% |
| retest2 (combined) | 126 | 44 | 34.9% | 4 | 112 | 10 | 0.18% | 22.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 13:15:00 | 662.75 | 665.04 | 665.13 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 11:15:00 | 669.20 | 665.58 | 665.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 14:15:00 | 669.60 | 666.21 | 665.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 671.85 | 672.48 | 670.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 671.85 | 672.48 | 670.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 671.85 | 672.48 | 670.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 671.85 | 672.48 | 670.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 672.70 | 672.52 | 670.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 672.70 | 672.52 | 670.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 670.65 | 672.15 | 670.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 670.65 | 672.15 | 670.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 668.50 | 671.42 | 670.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 668.50 | 671.42 | 670.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 668.15 | 670.77 | 670.20 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 665.15 | 669.20 | 669.56 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 11:15:00 | 674.05 | 670.12 | 669.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 14:15:00 | 675.60 | 671.56 | 670.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 667.65 | 671.01 | 670.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 667.65 | 671.01 | 670.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 667.65 | 671.01 | 670.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 667.65 | 671.01 | 670.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 666.00 | 670.00 | 670.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 645.70 | 662.08 | 665.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 12:15:00 | 656.15 | 645.22 | 650.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 656.15 | 645.22 | 650.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 656.15 | 645.22 | 650.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:00:00 | 656.15 | 645.22 | 650.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 668.75 | 649.92 | 652.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 668.75 | 649.92 | 652.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 664.00 | 655.36 | 654.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 10:15:00 | 665.00 | 657.29 | 655.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 13:15:00 | 654.95 | 657.86 | 656.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 654.95 | 657.86 | 656.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 654.95 | 657.86 | 656.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 654.95 | 657.86 | 656.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 653.25 | 656.93 | 656.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 652.65 | 656.93 | 656.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 654.00 | 656.35 | 655.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 655.55 | 656.35 | 655.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 648.65 | 654.81 | 655.23 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 10:15:00 | 661.25 | 656.10 | 655.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 12:15:00 | 668.00 | 658.78 | 657.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 10:15:00 | 664.80 | 665.84 | 661.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-31 10:30:00 | 666.45 | 665.84 | 661.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 664.00 | 665.10 | 662.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 13:30:00 | 666.15 | 665.72 | 662.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 655.55 | 663.68 | 662.08 | SL hit (close<static) qty=1.00 sl=662.05 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 649.25 | 663.22 | 663.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 645.20 | 659.62 | 662.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 670.00 | 661.70 | 662.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 670.00 | 661.70 | 662.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 670.00 | 661.70 | 662.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 670.00 | 661.70 | 662.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 13:15:00 | 684.60 | 666.28 | 664.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 14:15:00 | 691.25 | 671.27 | 667.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 720.40 | 723.66 | 706.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 11:45:00 | 722.95 | 723.66 | 706.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 721.30 | 721.17 | 711.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:30:00 | 715.00 | 721.17 | 711.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 724.00 | 721.54 | 714.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 10:15:00 | 726.15 | 721.82 | 716.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 15:00:00 | 726.65 | 724.12 | 719.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 728.80 | 724.39 | 720.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 749.05 | 755.89 | 756.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 749.05 | 755.89 | 756.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 747.50 | 754.05 | 755.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 761.70 | 755.30 | 755.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 11:15:00 | 761.70 | 755.30 | 755.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 761.70 | 755.30 | 755.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:45:00 | 757.00 | 755.30 | 755.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 12:15:00 | 760.35 | 756.31 | 755.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 14:15:00 | 764.00 | 757.63 | 756.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 15:15:00 | 770.05 | 770.11 | 765.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:15:00 | 772.30 | 770.11 | 765.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 757.50 | 772.26 | 770.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 757.50 | 772.26 | 770.18 | SL hit (close<ema400) qty=1.00 sl=770.18 alert=retest1 |

### Cycle 13 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 758.70 | 767.70 | 768.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 755.30 | 763.05 | 765.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 758.30 | 752.92 | 757.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 758.30 | 752.92 | 757.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 758.30 | 752.92 | 757.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 758.30 | 752.92 | 757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 758.00 | 753.94 | 757.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 758.00 | 753.94 | 757.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 770.30 | 757.21 | 758.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 770.30 | 757.21 | 758.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 769.50 | 759.67 | 759.44 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 14:15:00 | 753.25 | 759.33 | 760.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 751.05 | 755.56 | 757.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 762.30 | 756.50 | 757.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 762.30 | 756.50 | 757.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 762.30 | 756.50 | 757.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 764.70 | 756.50 | 757.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 768.60 | 758.92 | 758.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 780.95 | 763.32 | 760.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 775.50 | 777.09 | 770.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 775.50 | 777.09 | 770.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 782.75 | 789.67 | 783.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 784.70 | 789.67 | 783.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 780.40 | 787.81 | 783.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 781.35 | 787.81 | 783.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 784.70 | 787.19 | 783.67 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 777.90 | 782.02 | 782.37 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 785.50 | 782.98 | 782.76 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 775.35 | 781.63 | 782.29 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 794.90 | 784.49 | 783.15 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 773.95 | 785.78 | 785.88 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 11:15:00 | 803.00 | 785.44 | 784.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 14:15:00 | 812.95 | 796.21 | 790.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 15:15:00 | 814.90 | 815.25 | 808.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 09:15:00 | 813.35 | 815.25 | 808.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 815.40 | 814.72 | 809.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:30:00 | 811.00 | 814.72 | 809.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 811.65 | 813.45 | 809.79 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 795.85 | 808.00 | 808.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 788.50 | 798.20 | 802.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 784.00 | 782.85 | 788.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 11:00:00 | 784.00 | 782.85 | 788.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 787.75 | 782.44 | 786.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 790.50 | 782.44 | 786.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 784.15 | 782.78 | 785.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 787.55 | 785.52 | 786.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 797.55 | 787.92 | 787.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 802.40 | 790.82 | 789.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 822.90 | 823.94 | 815.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 839.15 | 823.94 | 815.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:15:00 | 881.11 | 865.55 | 846.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 889.00 | 895.80 | 885.40 | SL hit (close<ema200) qty=0.50 sl=895.80 alert=retest1 |

### Cycle 25 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 881.95 | 888.13 | 888.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 871.25 | 881.69 | 884.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 880.00 | 878.59 | 882.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:30:00 | 879.55 | 878.59 | 882.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 884.30 | 879.68 | 882.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 884.70 | 879.68 | 882.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 884.90 | 880.72 | 882.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 883.55 | 880.72 | 882.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 887.75 | 882.13 | 882.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 887.75 | 882.13 | 882.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 884.55 | 882.02 | 882.69 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 890.40 | 883.69 | 883.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 10:15:00 | 896.90 | 888.06 | 885.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 837.80 | 889.62 | 889.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 837.80 | 889.62 | 889.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 837.80 | 889.62 | 889.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 837.80 | 889.62 | 889.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 830.00 | 877.70 | 884.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 828.20 | 867.80 | 878.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 13:15:00 | 811.00 | 809.72 | 835.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:00:00 | 811.00 | 809.72 | 835.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 813.30 | 809.06 | 817.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 813.30 | 809.06 | 817.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 818.00 | 810.85 | 817.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 818.00 | 810.85 | 817.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 823.00 | 813.28 | 817.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:45:00 | 817.65 | 816.93 | 818.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:15:00 | 815.25 | 816.93 | 818.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 14:45:00 | 816.35 | 816.36 | 817.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:15:00 | 776.77 | 784.20 | 789.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:15:00 | 775.53 | 784.20 | 789.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 13:15:00 | 774.49 | 780.48 | 787.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 784.00 | 778.51 | 784.30 | SL hit (close>ema200) qty=0.50 sl=778.51 alert=retest2 |

### Cycle 28 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 792.80 | 787.41 | 786.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 12:15:00 | 795.70 | 791.73 | 789.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 813.95 | 816.16 | 806.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 15:00:00 | 813.95 | 816.16 | 806.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 813.60 | 815.30 | 807.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 813.60 | 815.30 | 807.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 810.55 | 814.35 | 808.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:30:00 | 811.75 | 814.35 | 808.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 804.30 | 812.34 | 807.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 804.30 | 812.34 | 807.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 801.65 | 810.20 | 807.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:45:00 | 801.60 | 810.20 | 807.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 804.00 | 808.96 | 806.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:45:00 | 807.00 | 809.02 | 807.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 811.10 | 812.89 | 813.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 811.10 | 812.89 | 813.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 11:15:00 | 807.35 | 811.79 | 812.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 11:15:00 | 805.25 | 804.87 | 807.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 11:45:00 | 803.95 | 804.87 | 807.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 807.40 | 804.93 | 807.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:45:00 | 807.15 | 804.93 | 807.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 807.90 | 805.53 | 807.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:30:00 | 807.00 | 805.53 | 807.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 803.60 | 805.14 | 807.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 808.40 | 805.14 | 807.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 810.70 | 806.25 | 807.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 811.55 | 806.25 | 807.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 805.55 | 806.11 | 807.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:30:00 | 799.55 | 805.22 | 806.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 12:00:00 | 801.05 | 799.61 | 800.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 13:15:00 | 805.35 | 801.36 | 801.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 13:15:00 | 805.35 | 801.36 | 801.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 14:15:00 | 811.15 | 803.32 | 802.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 14:15:00 | 809.40 | 810.51 | 807.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 15:00:00 | 809.40 | 810.51 | 807.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 805.60 | 809.46 | 807.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 805.60 | 809.46 | 807.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 804.50 | 808.47 | 807.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:45:00 | 804.75 | 808.47 | 807.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 804.15 | 807.60 | 806.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 12:30:00 | 809.50 | 808.04 | 807.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 807.15 | 806.73 | 806.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 797.60 | 804.97 | 805.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 797.60 | 804.97 | 805.88 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 809.00 | 806.39 | 806.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 821.75 | 809.47 | 807.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 834.50 | 834.83 | 826.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:45:00 | 835.70 | 834.83 | 826.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 826.85 | 831.71 | 827.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 826.85 | 831.71 | 827.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 821.30 | 829.62 | 826.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 821.30 | 829.62 | 826.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 825.05 | 828.71 | 826.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 828.15 | 828.71 | 826.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 825.70 | 827.77 | 826.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 825.70 | 827.77 | 826.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 825.65 | 827.35 | 826.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:00:00 | 825.65 | 827.35 | 826.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 825.50 | 826.98 | 826.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:45:00 | 825.55 | 826.98 | 826.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 825.25 | 826.63 | 826.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 825.05 | 826.63 | 826.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 825.00 | 826.31 | 826.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:15:00 | 827.00 | 826.31 | 826.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 827.00 | 826.44 | 826.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 832.90 | 826.44 | 826.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 819.25 | 825.01 | 825.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 819.25 | 825.01 | 825.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 812.55 | 822.51 | 824.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 11:15:00 | 811.25 | 809.74 | 815.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 12:00:00 | 811.25 | 809.74 | 815.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 811.70 | 810.81 | 814.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:45:00 | 807.50 | 809.67 | 813.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 817.65 | 811.82 | 813.81 | SL hit (close>static) qty=1.00 sl=815.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 822.25 | 815.38 | 815.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 829.75 | 820.10 | 817.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 11:15:00 | 826.65 | 831.85 | 828.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 11:15:00 | 826.65 | 831.85 | 828.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 826.65 | 831.85 | 828.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:45:00 | 827.20 | 831.85 | 828.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 824.75 | 830.43 | 828.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:45:00 | 824.55 | 830.43 | 828.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 830.00 | 830.33 | 828.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:45:00 | 836.90 | 831.29 | 829.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 837.20 | 831.29 | 829.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 837.50 | 837.45 | 833.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 10:00:00 | 837.45 | 837.46 | 834.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 833.15 | 836.60 | 834.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:45:00 | 833.20 | 836.60 | 834.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 835.50 | 836.38 | 834.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:30:00 | 834.10 | 836.38 | 834.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 836.95 | 836.49 | 834.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 836.95 | 836.49 | 834.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 834.00 | 835.99 | 834.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 834.40 | 835.99 | 834.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 833.60 | 835.51 | 834.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:15:00 | 833.50 | 835.51 | 834.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 833.50 | 835.11 | 834.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 835.35 | 835.11 | 834.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 819.50 | 836.56 | 837.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 819.50 | 836.56 | 837.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 818.00 | 832.85 | 835.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 827.00 | 826.12 | 830.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:45:00 | 826.85 | 826.12 | 830.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 797.90 | 797.42 | 804.16 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 809.45 | 806.82 | 806.48 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 803.00 | 806.05 | 806.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 799.15 | 804.67 | 805.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 788.85 | 787.81 | 791.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 13:45:00 | 787.90 | 787.81 | 791.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 779.55 | 785.90 | 789.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 10:00:00 | 777.60 | 780.85 | 784.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 12:00:00 | 777.75 | 779.55 | 783.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 777.90 | 779.08 | 782.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 738.72 | 751.50 | 755.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 738.86 | 751.50 | 755.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 739.00 | 751.50 | 755.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 14:15:00 | 739.55 | 738.99 | 744.32 | SL hit (close>ema200) qty=0.50 sl=738.99 alert=retest2 |

### Cycle 38 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 743.75 | 733.90 | 733.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 746.25 | 737.72 | 735.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 737.70 | 743.58 | 741.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 11:15:00 | 737.70 | 743.58 | 741.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 737.70 | 743.58 | 741.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 737.70 | 743.58 | 741.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 736.60 | 742.18 | 740.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 735.55 | 742.18 | 740.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 738.65 | 741.48 | 740.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:30:00 | 737.70 | 741.48 | 740.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 743.35 | 747.42 | 743.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 743.35 | 747.42 | 743.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 762.00 | 750.34 | 745.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:15:00 | 732.50 | 750.34 | 745.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 738.00 | 747.87 | 744.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 734.80 | 747.87 | 744.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 736.60 | 745.61 | 744.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 733.95 | 745.61 | 744.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 732.55 | 741.14 | 742.22 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 744.60 | 742.82 | 742.68 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 738.40 | 741.94 | 742.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 731.30 | 739.81 | 741.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 740.40 | 738.58 | 740.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 740.40 | 738.58 | 740.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 740.40 | 738.58 | 740.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 740.40 | 738.58 | 740.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 744.90 | 739.84 | 740.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:45:00 | 745.60 | 739.84 | 740.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 751.55 | 742.18 | 741.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 756.75 | 745.10 | 743.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 760.85 | 761.35 | 755.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 760.85 | 761.35 | 755.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 753.50 | 759.56 | 756.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 753.50 | 759.56 | 756.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 756.50 | 758.95 | 756.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 756.50 | 758.95 | 756.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 753.85 | 757.93 | 755.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 753.85 | 757.93 | 755.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 753.15 | 756.97 | 755.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:30:00 | 755.45 | 755.93 | 755.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 744.90 | 753.03 | 754.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 744.90 | 753.03 | 754.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 15:15:00 | 741.00 | 746.48 | 749.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 15:15:00 | 727.65 | 725.38 | 731.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:15:00 | 725.50 | 725.38 | 731.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 747.20 | 729.74 | 733.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 747.20 | 729.74 | 733.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 741.65 | 732.12 | 734.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 748.50 | 732.12 | 734.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 734.45 | 731.65 | 733.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:00:00 | 734.45 | 731.65 | 733.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 731.45 | 731.61 | 733.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 726.95 | 732.25 | 733.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 743.85 | 730.51 | 730.93 | SL hit (close>static) qty=1.00 sl=735.90 alert=retest2 |

### Cycle 44 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 743.20 | 733.05 | 732.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 747.85 | 737.85 | 734.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 730.25 | 737.73 | 735.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 15:15:00 | 730.25 | 737.73 | 735.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 730.25 | 737.73 | 735.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 733.90 | 736.26 | 734.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 728.65 | 734.74 | 734.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 726.15 | 734.74 | 734.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 728.60 | 733.51 | 733.89 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 747.40 | 735.21 | 734.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 15:15:00 | 748.00 | 740.00 | 736.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 755.75 | 757.67 | 751.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 755.75 | 757.67 | 751.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 751.95 | 756.18 | 751.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:45:00 | 752.10 | 756.18 | 751.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 751.00 | 755.14 | 751.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 751.00 | 755.14 | 751.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 751.45 | 754.40 | 751.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 742.00 | 754.40 | 751.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 747.30 | 752.98 | 751.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 744.50 | 752.98 | 751.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 753.80 | 753.15 | 751.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:45:00 | 757.60 | 754.01 | 752.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:00:00 | 758.00 | 754.81 | 752.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 756.25 | 753.32 | 752.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 745.00 | 751.64 | 751.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 745.00 | 751.64 | 751.81 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 757.35 | 751.64 | 750.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 15:15:00 | 761.80 | 753.67 | 751.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 12:15:00 | 754.80 | 754.88 | 753.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 13:00:00 | 754.80 | 754.88 | 753.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 761.20 | 756.98 | 754.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:15:00 | 766.35 | 756.98 | 754.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 764.50 | 758.69 | 755.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 760.35 | 765.27 | 765.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 760.35 | 765.27 | 765.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 757.15 | 761.96 | 764.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 735.20 | 725.29 | 731.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 735.20 | 725.29 | 731.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 735.20 | 725.29 | 731.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 735.20 | 725.29 | 731.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 733.55 | 726.94 | 731.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:45:00 | 734.65 | 726.94 | 731.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 734.90 | 729.65 | 731.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:45:00 | 734.95 | 729.65 | 731.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 737.65 | 733.74 | 733.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 745.05 | 736.00 | 734.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 734.60 | 736.20 | 734.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 11:15:00 | 734.60 | 736.20 | 734.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 734.60 | 736.20 | 734.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 734.60 | 736.20 | 734.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 733.95 | 735.75 | 734.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 735.35 | 735.75 | 734.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 732.50 | 735.10 | 734.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:45:00 | 734.65 | 735.10 | 734.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 733.15 | 734.71 | 734.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 733.45 | 734.71 | 734.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 731.05 | 733.98 | 734.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 724.50 | 732.08 | 733.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 709.90 | 706.51 | 712.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 709.90 | 706.51 | 712.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 707.05 | 706.62 | 711.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:30:00 | 711.40 | 706.62 | 711.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 711.05 | 707.73 | 711.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:30:00 | 711.50 | 707.73 | 711.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 706.25 | 707.44 | 710.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:30:00 | 713.00 | 707.44 | 710.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 690.55 | 687.77 | 693.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 690.55 | 687.77 | 693.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 695.05 | 689.22 | 693.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 695.05 | 689.22 | 693.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 687.30 | 688.84 | 693.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 15:00:00 | 686.40 | 688.83 | 692.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:45:00 | 686.60 | 687.25 | 690.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 690.70 | 687.24 | 686.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 690.70 | 687.24 | 686.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 15:15:00 | 693.30 | 689.36 | 688.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 688.10 | 689.11 | 688.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 688.10 | 689.11 | 688.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 688.10 | 689.11 | 688.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 688.10 | 689.11 | 688.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 689.90 | 689.27 | 688.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 691.00 | 689.27 | 688.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:45:00 | 690.85 | 690.26 | 688.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 681.70 | 688.75 | 689.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 681.70 | 688.75 | 689.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 670.25 | 685.05 | 687.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 10:15:00 | 667.80 | 667.69 | 672.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:45:00 | 663.65 | 666.99 | 671.73 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 670.15 | 667.53 | 670.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:45:00 | 668.40 | 667.53 | 670.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 678.25 | 669.73 | 671.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 678.25 | 669.73 | 671.21 | SL hit (close>ema400) qty=1.00 sl=671.21 alert=retest1 |

### Cycle 54 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 652.90 | 647.39 | 646.85 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 642.65 | 647.16 | 647.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 640.50 | 645.83 | 646.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 634.80 | 632.35 | 636.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 634.80 | 632.35 | 636.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 640.50 | 632.49 | 635.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 640.50 | 632.49 | 635.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 636.55 | 633.30 | 635.15 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 640.10 | 636.15 | 636.12 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 635.00 | 635.96 | 636.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 10:15:00 | 631.95 | 635.20 | 635.69 | Break + close below crossover candle low |

### Cycle 58 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 647.10 | 637.26 | 636.52 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 629.85 | 635.78 | 635.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 620.65 | 631.85 | 634.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 625.95 | 622.38 | 626.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 625.95 | 622.38 | 626.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 625.95 | 622.38 | 626.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 625.00 | 622.38 | 626.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 622.60 | 621.99 | 624.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 622.10 | 621.99 | 624.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 628.05 | 623.20 | 625.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 619.40 | 623.20 | 625.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:30:00 | 620.65 | 620.44 | 621.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 13:15:00 | 633.15 | 624.08 | 622.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 633.15 | 624.08 | 622.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 634.70 | 627.84 | 625.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 636.30 | 636.41 | 632.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:00:00 | 636.30 | 636.41 | 632.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 633.75 | 636.05 | 633.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 630.00 | 636.05 | 633.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 641.85 | 637.21 | 634.29 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 605.30 | 628.08 | 631.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 584.80 | 593.17 | 600.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 585.35 | 583.58 | 589.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 585.35 | 583.58 | 589.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 579.55 | 583.09 | 588.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:45:00 | 578.15 | 581.03 | 586.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 578.85 | 580.49 | 584.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:45:00 | 577.35 | 580.02 | 583.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 12:00:00 | 578.15 | 579.64 | 583.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 580.75 | 578.17 | 579.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:45:00 | 579.80 | 578.17 | 579.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 579.90 | 578.52 | 579.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:30:00 | 580.55 | 578.52 | 579.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 578.95 | 578.61 | 579.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 582.40 | 578.61 | 579.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 576.30 | 578.14 | 579.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 11:15:00 | 574.30 | 577.52 | 579.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 569.20 | 575.82 | 577.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 12:15:00 | 583.00 | 578.01 | 577.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 12:15:00 | 583.00 | 578.01 | 577.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 09:15:00 | 591.65 | 581.57 | 579.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 11:15:00 | 598.90 | 600.10 | 592.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-27 11:30:00 | 600.00 | 600.10 | 592.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 583.00 | 595.99 | 591.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:00:00 | 583.00 | 595.99 | 591.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 577.30 | 592.25 | 590.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:45:00 | 573.45 | 592.25 | 590.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 15:15:00 | 577.50 | 589.30 | 589.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 563.85 | 584.21 | 587.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 563.60 | 560.44 | 568.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:00:00 | 563.60 | 560.44 | 568.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 559.15 | 560.18 | 567.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:15:00 | 558.50 | 561.21 | 566.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 09:15:00 | 551.50 | 562.02 | 564.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 09:45:00 | 557.40 | 561.18 | 564.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 570.50 | 565.45 | 565.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 570.50 | 565.45 | 565.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 574.80 | 567.40 | 566.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 567.45 | 568.00 | 566.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 567.45 | 568.00 | 566.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 567.45 | 567.89 | 566.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:45:00 | 567.10 | 567.89 | 566.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 566.00 | 567.51 | 566.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:45:00 | 564.60 | 567.51 | 566.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 565.85 | 567.18 | 566.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:45:00 | 565.10 | 567.18 | 566.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 566.00 | 566.94 | 566.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 571.05 | 566.59 | 566.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 566.30 | 567.68 | 567.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 562.15 | 566.57 | 566.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 562.15 | 566.57 | 566.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 560.25 | 565.31 | 566.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 570.65 | 564.30 | 565.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 570.65 | 564.30 | 565.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 570.65 | 564.30 | 565.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 570.65 | 564.30 | 565.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 10:15:00 | 573.70 | 566.18 | 565.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 12:15:00 | 578.50 | 570.21 | 567.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 573.80 | 574.39 | 570.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-12 10:00:00 | 573.80 | 574.39 | 570.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 572.85 | 574.09 | 571.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:15:00 | 570.40 | 574.09 | 571.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 567.35 | 572.74 | 570.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 567.35 | 572.74 | 570.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 570.45 | 572.28 | 570.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 15:00:00 | 572.80 | 572.16 | 570.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 572.40 | 573.82 | 573.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 10:00:00 | 571.50 | 573.36 | 572.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 569.75 | 572.64 | 572.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 569.75 | 572.64 | 572.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 566.70 | 570.87 | 571.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 571.15 | 570.92 | 571.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 14:00:00 | 571.15 | 570.92 | 571.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 568.00 | 570.19 | 571.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 569.10 | 570.19 | 571.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 570.45 | 570.24 | 571.20 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 575.85 | 571.53 | 571.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 578.30 | 572.89 | 572.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 10:15:00 | 651.90 | 656.68 | 647.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 11:00:00 | 651.90 | 656.68 | 647.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 648.50 | 654.67 | 648.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 648.50 | 654.67 | 648.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 648.75 | 653.49 | 648.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 648.90 | 653.49 | 648.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 646.65 | 652.12 | 648.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 646.65 | 652.12 | 648.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 648.50 | 651.40 | 648.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 649.10 | 651.40 | 648.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 645.65 | 650.25 | 647.85 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 13:15:00 | 636.25 | 644.70 | 645.81 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 669.40 | 649.64 | 647.95 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 644.65 | 648.46 | 648.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 13:15:00 | 644.05 | 647.06 | 648.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 657.25 | 642.53 | 643.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 14:15:00 | 657.25 | 642.53 | 643.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 657.25 | 642.53 | 643.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 657.25 | 642.53 | 643.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 632.95 | 640.62 | 642.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 631.20 | 640.17 | 641.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 12:00:00 | 631.00 | 637.07 | 639.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:00:00 | 631.05 | 635.87 | 638.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:30:00 | 629.55 | 634.89 | 638.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 568.08 | 625.66 | 633.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 646.50 | 632.41 | 631.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 652.80 | 640.94 | 635.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 10:15:00 | 645.65 | 646.65 | 640.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-09 11:00:00 | 645.65 | 646.65 | 640.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 672.20 | 678.38 | 674.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:30:00 | 670.30 | 678.38 | 674.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 671.65 | 677.03 | 673.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:30:00 | 671.40 | 677.03 | 673.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 09:15:00 | 663.45 | 671.47 | 671.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 13:15:00 | 658.75 | 666.55 | 669.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 14:15:00 | 656.00 | 655.93 | 659.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-23 15:00:00 | 656.00 | 655.93 | 659.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 657.75 | 656.43 | 658.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 11:15:00 | 656.60 | 656.84 | 658.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:30:00 | 649.50 | 655.43 | 657.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 13:15:00 | 623.77 | 633.67 | 639.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 15:15:00 | 617.02 | 629.41 | 636.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-02 14:15:00 | 629.60 | 625.71 | 630.92 | SL hit (close>ema200) qty=0.50 sl=625.71 alert=retest2 |

### Cycle 74 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 615.45 | 603.69 | 602.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 618.85 | 613.95 | 609.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 636.05 | 637.39 | 632.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 636.05 | 637.39 | 632.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 641.90 | 637.91 | 633.84 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 13:15:00 | 633.45 | 636.41 | 636.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 15:15:00 | 631.00 | 634.85 | 635.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 635.75 | 634.96 | 635.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 635.75 | 634.96 | 635.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 635.75 | 634.96 | 635.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 635.75 | 634.96 | 635.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 635.30 | 635.03 | 635.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 631.95 | 635.03 | 635.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 12:15:00 | 634.25 | 634.14 | 634.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:00:00 | 632.65 | 633.85 | 634.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 637.05 | 634.34 | 634.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 637.05 | 634.34 | 634.28 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 633.55 | 634.18 | 634.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 630.80 | 633.50 | 633.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 638.05 | 633.65 | 633.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 638.05 | 633.65 | 633.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 638.05 | 633.65 | 633.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 638.05 | 633.65 | 633.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 635.40 | 634.00 | 633.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 643.60 | 637.47 | 635.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 635.90 | 638.02 | 636.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 635.90 | 638.02 | 636.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 635.90 | 638.02 | 636.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 634.90 | 638.02 | 636.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 637.35 | 637.88 | 636.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:15:00 | 635.00 | 637.88 | 636.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 632.05 | 636.72 | 636.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 632.45 | 636.72 | 636.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 632.10 | 635.79 | 635.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 630.75 | 634.78 | 635.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 635.85 | 632.96 | 634.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 635.85 | 632.96 | 634.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 635.85 | 632.96 | 634.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:45:00 | 637.00 | 632.96 | 634.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 638.00 | 633.97 | 634.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 638.00 | 633.97 | 634.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 635.35 | 634.25 | 634.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 633.00 | 634.25 | 634.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 14:15:00 | 637.90 | 635.10 | 634.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 637.90 | 635.10 | 634.95 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 632.30 | 634.48 | 634.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 631.95 | 633.97 | 634.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 633.70 | 633.67 | 634.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 13:15:00 | 633.70 | 633.67 | 634.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 633.70 | 633.67 | 634.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:00:00 | 633.70 | 633.67 | 634.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 633.15 | 633.14 | 633.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 628.00 | 630.12 | 631.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 14:15:00 | 626.55 | 623.29 | 623.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 626.55 | 623.29 | 623.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 638.55 | 626.65 | 624.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 627.50 | 628.18 | 626.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 627.50 | 628.18 | 626.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 627.50 | 628.18 | 626.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:15:00 | 625.95 | 628.18 | 626.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 625.95 | 627.73 | 626.39 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 624.00 | 625.48 | 625.60 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 635.85 | 627.08 | 626.22 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 623.05 | 627.61 | 627.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 618.45 | 624.51 | 626.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 620.80 | 620.01 | 622.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 620.80 | 620.01 | 622.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 622.85 | 620.71 | 622.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 622.85 | 620.71 | 622.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 623.90 | 621.35 | 622.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 622.60 | 621.35 | 622.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 627.20 | 622.52 | 623.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 627.80 | 622.52 | 623.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 626.65 | 623.35 | 623.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 624.50 | 623.35 | 623.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 629.85 | 624.65 | 624.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 629.85 | 624.65 | 624.10 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 618.40 | 623.87 | 624.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 614.50 | 619.33 | 621.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 606.35 | 602.73 | 608.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 606.35 | 602.73 | 608.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 606.35 | 602.73 | 608.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 606.35 | 602.73 | 608.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 605.75 | 603.33 | 608.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 600.85 | 603.33 | 608.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 605.10 | 603.49 | 607.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 12:15:00 | 598.95 | 595.86 | 595.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 598.95 | 595.86 | 595.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 600.60 | 597.92 | 596.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 597.65 | 598.11 | 597.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 597.65 | 598.11 | 597.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 597.65 | 598.11 | 597.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 597.65 | 598.11 | 597.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 598.85 | 598.26 | 597.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:30:00 | 599.40 | 598.48 | 597.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 595.25 | 597.84 | 597.46 | SL hit (close<static) qty=1.00 sl=597.05 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 594.45 | 597.16 | 597.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 586.10 | 593.38 | 595.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 590.50 | 590.16 | 592.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 590.50 | 590.16 | 592.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 589.70 | 590.04 | 592.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 584.75 | 589.77 | 590.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 573.40 | 570.01 | 569.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 573.40 | 570.01 | 569.60 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 12:15:00 | 567.15 | 569.45 | 569.60 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 571.60 | 569.93 | 569.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 575.55 | 571.40 | 570.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 572.05 | 574.89 | 573.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 572.05 | 574.89 | 573.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 572.05 | 574.89 | 573.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 572.05 | 574.89 | 573.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 573.10 | 574.53 | 573.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 571.50 | 574.53 | 573.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 575.95 | 573.77 | 573.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:15:00 | 573.20 | 573.77 | 573.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 573.20 | 573.66 | 573.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 572.65 | 573.66 | 573.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 574.80 | 573.89 | 573.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:00:00 | 579.20 | 575.33 | 574.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 582.40 | 585.55 | 585.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 582.40 | 585.55 | 585.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 578.00 | 583.10 | 584.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 569.50 | 567.80 | 572.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 569.50 | 567.80 | 572.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 574.00 | 569.04 | 572.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 574.00 | 569.04 | 572.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 573.95 | 570.02 | 572.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 574.50 | 570.02 | 572.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 572.35 | 570.49 | 572.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 568.35 | 570.73 | 571.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 567.55 | 558.15 | 557.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 567.55 | 558.15 | 557.12 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 577.95 | 579.88 | 580.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 575.50 | 578.56 | 579.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 13:15:00 | 547.35 | 545.56 | 552.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 14:00:00 | 547.35 | 545.56 | 552.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 545.55 | 545.17 | 550.86 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 571.70 | 550.83 | 550.83 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 557.85 | 560.25 | 560.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 555.75 | 558.84 | 559.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 559.00 | 558.87 | 559.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 559.10 | 558.87 | 559.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 558.15 | 558.17 | 559.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:45:00 | 552.85 | 557.14 | 558.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 554.80 | 554.41 | 556.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 554.80 | 554.62 | 556.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 564.00 | 557.80 | 557.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 564.00 | 557.80 | 557.17 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 555.00 | 557.16 | 557.26 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 562.80 | 558.29 | 557.76 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 555.95 | 559.15 | 559.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 553.75 | 558.07 | 558.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 555.00 | 554.71 | 556.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 555.00 | 554.71 | 556.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 555.00 | 554.71 | 556.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 557.80 | 554.71 | 556.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 556.35 | 555.04 | 556.29 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 560.50 | 556.78 | 556.69 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 554.70 | 558.54 | 558.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 553.65 | 556.11 | 557.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 528.50 | 526.58 | 531.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 14:15:00 | 528.50 | 526.58 | 531.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 528.50 | 526.58 | 531.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 533.15 | 526.58 | 531.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 525.50 | 526.36 | 530.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 522.00 | 525.49 | 529.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 522.00 | 523.94 | 527.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 545.30 | 527.92 | 528.40 | SL hit (close>static) qty=1.00 sl=535.60 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 556.95 | 533.73 | 530.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 567.15 | 544.42 | 536.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 564.60 | 567.43 | 553.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 11:00:00 | 564.60 | 567.43 | 553.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 571.60 | 577.74 | 574.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 571.60 | 577.74 | 574.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 572.00 | 576.59 | 574.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 572.00 | 576.59 | 574.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 572.00 | 575.67 | 573.95 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 569.00 | 572.70 | 572.93 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 578.20 | 573.20 | 572.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 581.45 | 577.15 | 575.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 577.10 | 577.97 | 576.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 577.10 | 577.97 | 576.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 577.10 | 577.97 | 576.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 577.10 | 577.97 | 576.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 574.05 | 577.18 | 576.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 574.75 | 577.18 | 576.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 573.55 | 576.46 | 576.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 573.70 | 576.46 | 576.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 573.25 | 575.82 | 575.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 570.50 | 574.75 | 575.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 570.30 | 568.29 | 570.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 570.30 | 568.29 | 570.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 570.30 | 568.29 | 570.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 569.35 | 568.29 | 570.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 570.90 | 569.50 | 570.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 570.90 | 569.50 | 570.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 570.20 | 569.64 | 570.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:15:00 | 571.00 | 569.64 | 570.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 571.00 | 569.91 | 570.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 567.25 | 569.91 | 570.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 564.75 | 568.88 | 570.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:30:00 | 563.15 | 567.70 | 569.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 574.45 | 568.57 | 567.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 574.45 | 568.57 | 567.87 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 561.35 | 567.62 | 567.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 557.45 | 565.58 | 566.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 556.50 | 556.46 | 559.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 556.50 | 556.46 | 559.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 556.50 | 556.46 | 559.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 558.40 | 556.46 | 559.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 557.00 | 556.67 | 558.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 557.00 | 556.67 | 558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 560.25 | 557.39 | 558.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 560.25 | 557.39 | 558.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 555.30 | 556.97 | 558.35 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 570.10 | 560.30 | 559.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 571.35 | 562.51 | 560.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 567.20 | 567.41 | 564.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 15:15:00 | 572.30 | 566.58 | 564.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 572.30 | 567.72 | 565.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 563.80 | 566.94 | 565.20 | SL hit (close<ema400) qty=1.00 sl=565.20 alert=retest1 |

### Cycle 111 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 562.55 | 564.00 | 564.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 560.40 | 563.28 | 563.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 562.85 | 562.54 | 563.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 562.85 | 562.54 | 563.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 562.85 | 562.54 | 563.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 563.10 | 562.54 | 563.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 562.60 | 562.55 | 563.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:15:00 | 564.30 | 562.55 | 563.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 563.35 | 562.71 | 563.22 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 568.65 | 564.15 | 563.76 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 559.05 | 562.82 | 563.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 553.50 | 560.04 | 561.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 554.00 | 553.01 | 556.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:30:00 | 553.40 | 553.01 | 556.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 554.75 | 553.49 | 555.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 554.40 | 553.49 | 555.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 553.65 | 553.52 | 555.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:15:00 | 554.70 | 553.52 | 555.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 554.70 | 553.76 | 555.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 555.90 | 553.76 | 555.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 552.95 | 553.60 | 555.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 549.40 | 553.97 | 554.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 557.00 | 555.39 | 555.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 557.00 | 555.39 | 555.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 558.00 | 555.91 | 555.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 561.90 | 563.10 | 560.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 561.90 | 563.10 | 560.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 563.65 | 563.21 | 561.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 13:45:00 | 564.75 | 563.37 | 561.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 564.65 | 563.37 | 561.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 559.55 | 562.39 | 561.57 | SL hit (close<static) qty=1.00 sl=560.10 alert=retest2 |

### Cycle 115 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 560.80 | 565.43 | 565.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 559.10 | 563.52 | 564.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 13:15:00 | 560.00 | 559.72 | 561.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:00:00 | 560.00 | 559.72 | 561.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 542.15 | 540.51 | 542.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 539.65 | 540.51 | 542.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 540.95 | 540.60 | 542.47 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 550.00 | 543.20 | 542.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 555.00 | 547.01 | 544.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 554.00 | 560.13 | 555.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 554.00 | 560.13 | 555.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 554.00 | 560.13 | 555.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 554.00 | 560.13 | 555.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 556.30 | 559.36 | 555.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 557.00 | 559.36 | 555.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:30:00 | 557.05 | 558.78 | 556.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 553.15 | 555.20 | 555.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 553.15 | 555.20 | 555.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 551.40 | 554.38 | 554.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 558.60 | 554.40 | 554.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 558.60 | 554.40 | 554.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 558.60 | 554.40 | 554.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 558.60 | 554.40 | 554.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 553.05 | 554.13 | 554.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 552.25 | 553.60 | 554.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 555.70 | 554.63 | 554.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 555.70 | 554.63 | 554.60 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 553.10 | 554.32 | 554.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 15:15:00 | 549.00 | 553.05 | 553.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 12:15:00 | 552.40 | 552.23 | 553.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 12:15:00 | 552.40 | 552.23 | 553.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 552.40 | 552.23 | 553.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 553.45 | 552.23 | 553.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 552.65 | 552.31 | 553.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:45:00 | 552.30 | 552.31 | 553.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 553.20 | 552.49 | 553.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 553.20 | 552.49 | 553.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 551.10 | 552.21 | 552.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 549.90 | 552.21 | 552.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 550.25 | 549.66 | 550.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 550.95 | 549.66 | 550.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 550.85 | 550.39 | 550.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 552.30 | 550.77 | 551.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 552.30 | 550.77 | 551.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 552.00 | 551.02 | 551.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 552.00 | 551.02 | 551.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 548.30 | 546.53 | 548.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 548.30 | 546.53 | 548.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 549.50 | 547.12 | 548.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 549.50 | 547.12 | 548.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 549.05 | 547.51 | 548.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 548.65 | 547.51 | 548.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 557.10 | 549.43 | 549.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 557.10 | 549.43 | 549.45 | SL hit (close>static) qty=1.00 sl=555.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 552.00 | 549.94 | 549.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 561.00 | 556.29 | 553.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 565.30 | 570.38 | 565.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 565.30 | 570.38 | 565.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 565.30 | 570.38 | 565.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 565.30 | 570.38 | 565.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 564.45 | 569.19 | 565.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 564.10 | 569.19 | 565.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 563.05 | 567.97 | 565.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 563.05 | 567.97 | 565.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 560.95 | 566.56 | 564.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 560.95 | 566.56 | 564.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 556.40 | 563.26 | 563.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 554.30 | 560.16 | 562.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 550.95 | 550.34 | 553.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 550.95 | 550.34 | 553.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 550.95 | 550.34 | 553.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 556.60 | 550.34 | 553.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 544.35 | 548.05 | 551.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 551.00 | 548.05 | 551.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 551.40 | 548.19 | 550.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 553.20 | 548.19 | 550.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 551.95 | 548.94 | 550.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 552.65 | 548.94 | 550.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 553.00 | 550.40 | 551.12 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 558.65 | 552.60 | 552.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 559.80 | 553.97 | 552.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 574.60 | 576.54 | 571.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 15:15:00 | 570.00 | 574.43 | 570.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 570.00 | 574.43 | 570.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 568.10 | 574.43 | 570.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 566.95 | 572.94 | 570.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 566.95 | 572.94 | 570.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 562.10 | 570.77 | 569.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 562.10 | 570.77 | 569.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 562.85 | 569.19 | 569.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 558.90 | 567.13 | 568.26 | Break + close below crossover candle low |

### Cycle 124 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 583.70 | 570.13 | 569.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 587.70 | 577.71 | 573.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 573.00 | 578.76 | 574.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 573.00 | 578.76 | 574.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 573.00 | 578.76 | 574.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 573.00 | 578.76 | 574.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 574.20 | 577.85 | 574.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 580.40 | 579.15 | 575.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 582.70 | 584.24 | 584.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 582.70 | 584.24 | 584.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 577.15 | 582.58 | 583.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 567.50 | 564.32 | 570.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 567.50 | 564.32 | 570.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 567.50 | 564.32 | 570.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 556.10 | 565.26 | 568.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:00:00 | 558.10 | 563.83 | 567.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:45:00 | 557.90 | 562.07 | 565.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 557.15 | 555.19 | 557.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 558.40 | 555.83 | 557.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 555.65 | 556.66 | 557.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 554.00 | 556.78 | 557.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 555.60 | 555.56 | 556.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 555.40 | 555.60 | 556.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 554.20 | 555.32 | 556.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 556.85 | 555.32 | 556.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 547.80 | 553.77 | 555.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 537.00 | 549.50 | 552.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 552.90 | 548.28 | 547.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 14:15:00 | 552.90 | 548.28 | 547.78 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 13:15:00 | 540.75 | 547.55 | 547.89 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 550.00 | 548.12 | 547.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 553.80 | 550.25 | 549.13 | Break + close above crossover candle high |

### Cycle 129 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 529.90 | 551.56 | 551.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 525.30 | 546.31 | 549.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 538.75 | 535.29 | 540.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 11:00:00 | 538.75 | 535.29 | 540.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 540.50 | 537.73 | 540.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 540.50 | 537.73 | 540.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 542.00 | 538.58 | 540.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 537.90 | 538.58 | 540.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 538.80 | 536.51 | 538.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 532.45 | 537.00 | 538.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 542.10 | 538.02 | 538.49 | SL hit (close>static) qty=1.00 sl=542.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 546.85 | 539.79 | 539.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 555.50 | 544.56 | 541.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 550.80 | 550.89 | 547.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 550.80 | 550.89 | 547.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 545.85 | 549.90 | 547.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 534.80 | 549.90 | 547.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 536.40 | 547.20 | 546.32 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 537.00 | 545.16 | 545.47 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 552.95 | 545.11 | 545.04 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 543.05 | 546.17 | 546.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 533.80 | 542.86 | 544.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 535.90 | 535.50 | 538.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 535.90 | 535.50 | 538.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 554.70 | 539.74 | 539.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:15:00 | 555.25 | 539.74 | 539.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 558.00 | 543.39 | 541.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 564.05 | 550.00 | 544.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 608.65 | 608.86 | 596.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 10:30:00 | 605.80 | 608.86 | 596.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 601.00 | 605.40 | 599.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 593.40 | 605.40 | 599.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 588.10 | 601.94 | 598.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 588.10 | 601.94 | 598.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 586.70 | 598.89 | 597.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 586.70 | 598.89 | 597.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 585.15 | 596.14 | 596.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 576.35 | 588.84 | 592.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 578.55 | 576.52 | 582.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:30:00 | 576.60 | 576.52 | 582.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 588.10 | 578.71 | 582.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 588.10 | 578.71 | 582.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 593.00 | 581.57 | 583.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 593.00 | 581.57 | 583.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 589.45 | 585.79 | 585.30 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 583.10 | 586.40 | 586.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 582.70 | 585.66 | 586.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 581.75 | 581.13 | 583.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:30:00 | 580.00 | 581.13 | 583.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 582.50 | 581.54 | 583.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 582.75 | 581.54 | 583.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 583.50 | 581.93 | 583.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 583.50 | 581.93 | 583.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 584.10 | 582.37 | 583.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 584.50 | 582.37 | 583.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 580.75 | 582.04 | 583.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 586.25 | 582.04 | 583.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 585.00 | 582.64 | 583.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 581.95 | 582.61 | 583.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 581.95 | 582.61 | 583.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 581.55 | 582.41 | 583.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 586.90 | 584.08 | 583.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 586.90 | 584.08 | 583.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 594.40 | 587.72 | 586.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 633.20 | 634.04 | 620.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 626.80 | 632.48 | 624.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 626.80 | 632.48 | 624.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 626.80 | 632.48 | 624.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 624.00 | 629.76 | 624.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 602.90 | 629.76 | 624.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 601.10 | 624.02 | 622.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:30:00 | 598.95 | 624.02 | 622.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 592.75 | 617.77 | 619.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 15:15:00 | 591.20 | 602.46 | 610.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 596.80 | 595.94 | 603.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 596.15 | 595.94 | 603.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 604.25 | 593.88 | 598.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 604.25 | 593.88 | 598.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 600.25 | 595.15 | 598.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 596.10 | 595.15 | 598.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 566.29 | 580.29 | 582.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 582.30 | 580.69 | 582.32 | SL hit (close>ema200) qty=0.50 sl=580.69 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 592.50 | 583.44 | 583.31 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 578.30 | 583.45 | 583.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 573.95 | 581.55 | 583.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 558.35 | 557.00 | 566.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 558.25 | 557.00 | 566.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 565.15 | 558.41 | 565.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 565.15 | 558.41 | 565.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 569.55 | 560.64 | 565.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 569.55 | 560.64 | 565.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 567.90 | 562.09 | 565.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 565.90 | 562.09 | 565.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 566.65 | 564.31 | 566.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 565.95 | 564.64 | 566.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 582.50 | 568.91 | 568.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 582.50 | 568.91 | 568.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 586.45 | 572.42 | 569.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 11:15:00 | 593.00 | 597.81 | 591.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-20 12:00:00 | 593.00 | 597.81 | 591.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 592.00 | 596.65 | 591.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 592.00 | 596.65 | 591.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 593.60 | 596.04 | 591.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 592.05 | 596.04 | 591.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 593.05 | 595.44 | 591.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:30:00 | 592.35 | 595.44 | 591.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 593.90 | 595.13 | 592.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 598.40 | 595.13 | 592.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 600.60 | 596.23 | 592.93 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 582.90 | 592.85 | 593.41 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 605.60 | 595.37 | 594.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 14:15:00 | 621.30 | 600.55 | 596.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 12:15:00 | 609.30 | 610.69 | 604.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 13:00:00 | 609.30 | 610.69 | 604.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 603.75 | 608.56 | 604.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 595.75 | 608.56 | 604.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 587.60 | 604.37 | 603.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 587.60 | 604.37 | 603.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 588.40 | 601.17 | 601.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 582.15 | 592.46 | 596.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 579.00 | 574.55 | 582.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 579.00 | 574.55 | 582.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 581.00 | 575.84 | 582.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 566.55 | 579.04 | 582.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 584.10 | 576.51 | 579.55 | SL hit (close>static) qty=1.00 sl=582.30 alert=retest2 |

### Cycle 146 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 602.90 | 581.79 | 581.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 609.40 | 595.35 | 591.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 592.60 | 598.85 | 595.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 592.60 | 598.85 | 595.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 592.60 | 598.85 | 595.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 592.60 | 598.85 | 595.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 595.05 | 598.09 | 595.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 597.40 | 597.60 | 595.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 15:15:00 | 589.15 | 594.08 | 594.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 589.15 | 594.08 | 594.31 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 607.85 | 596.83 | 595.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 626.15 | 605.10 | 600.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 14:15:00 | 620.70 | 622.75 | 615.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 15:00:00 | 620.70 | 622.75 | 615.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 649.00 | 647.06 | 640.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 651.00 | 647.06 | 640.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 650.60 | 648.78 | 642.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:45:00 | 650.45 | 647.10 | 644.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:45:00 | 655.40 | 648.53 | 645.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 637.85 | 647.00 | 645.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 637.85 | 647.00 | 645.91 | SL hit (close<static) qty=1.00 sl=640.15 alert=retest2 |

### Cycle 149 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 633.70 | 644.34 | 644.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 628.05 | 639.26 | 642.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 605.50 | 604.59 | 615.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:45:00 | 603.15 | 604.59 | 615.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 594.65 | 597.55 | 603.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:00:00 | 593.75 | 596.79 | 602.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:00:00 | 593.00 | 592.16 | 597.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 14:15:00 | 605.75 | 599.85 | 599.93 | SL hit (close>static) qty=1.00 sl=605.40 alert=retest2 |

### Cycle 150 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 609.00 | 601.68 | 600.75 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 556.45 | 592.63 | 596.73 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-31 13:30:00 | 666.15 | 2024-05-31 14:15:00 | 655.55 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-06-03 09:45:00 | 667.20 | 2024-06-04 09:15:00 | 653.15 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-06-10 10:15:00 | 726.15 | 2024-06-18 09:15:00 | 749.05 | STOP_HIT | 1.00 | 3.15% |
| BUY | retest2 | 2024-06-10 15:00:00 | 726.65 | 2024-06-18 09:15:00 | 749.05 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2024-06-11 09:15:00 | 728.80 | 2024-06-18 09:15:00 | 749.05 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest1 | 2024-06-21 09:15:00 | 772.30 | 2024-06-24 09:15:00 | 757.50 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest1 | 2024-07-29 09:15:00 | 839.15 | 2024-07-30 10:15:00 | 881.11 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-07-29 09:15:00 | 839.15 | 2024-08-01 13:15:00 | 889.00 | STOP_HIT | 0.50 | 5.94% |
| BUY | retest2 | 2024-08-02 10:30:00 | 903.80 | 2024-08-05 10:15:00 | 878.65 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-08-19 12:45:00 | 817.65 | 2024-08-23 11:15:00 | 776.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-19 13:15:00 | 815.25 | 2024-08-23 11:15:00 | 775.53 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2024-08-19 14:45:00 | 816.35 | 2024-08-23 13:15:00 | 774.49 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2024-08-19 12:45:00 | 817.65 | 2024-08-26 09:15:00 | 784.00 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2024-08-19 13:15:00 | 815.25 | 2024-08-26 09:15:00 | 784.00 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2024-08-19 14:45:00 | 816.35 | 2024-08-26 09:15:00 | 784.00 | STOP_HIT | 0.50 | 3.96% |
| BUY | retest2 | 2024-08-29 14:45:00 | 807.00 | 2024-09-03 10:15:00 | 811.10 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-09-05 12:30:00 | 799.55 | 2024-09-09 13:15:00 | 805.35 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-09 12:00:00 | 801.05 | 2024-09-09 13:15:00 | 805.35 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-09-11 12:30:00 | 809.50 | 2024-09-12 09:15:00 | 797.60 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-09-11 15:15:00 | 807.15 | 2024-09-12 09:15:00 | 797.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-19 09:15:00 | 832.90 | 2024-09-19 09:15:00 | 819.25 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-09-20 14:45:00 | 807.50 | 2024-09-23 10:15:00 | 817.65 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-27 09:45:00 | 836.90 | 2024-10-03 11:15:00 | 819.50 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-09-27 10:15:00 | 837.20 | 2024-10-03 11:15:00 | 819.50 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-09-27 15:15:00 | 837.50 | 2024-10-03 11:15:00 | 819.50 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-09-30 10:00:00 | 837.45 | 2024-10-03 11:15:00 | 819.50 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-10-01 09:15:00 | 835.35 | 2024-10-03 11:15:00 | 819.50 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-10-16 10:00:00 | 777.60 | 2024-10-23 09:15:00 | 738.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 12:00:00 | 777.75 | 2024-10-23 09:15:00 | 738.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 14:15:00 | 777.90 | 2024-10-23 09:15:00 | 739.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 10:00:00 | 777.60 | 2024-10-24 14:15:00 | 739.55 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2024-10-16 12:00:00 | 777.75 | 2024-10-24 14:15:00 | 739.55 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2024-10-16 14:15:00 | 777.90 | 2024-10-24 14:15:00 | 739.55 | STOP_HIT | 0.50 | 4.93% |
| BUY | retest2 | 2024-11-08 14:30:00 | 755.45 | 2024-11-11 09:15:00 | 744.90 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-11-18 09:15:00 | 726.95 | 2024-11-19 09:15:00 | 743.85 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-11-27 11:45:00 | 757.60 | 2024-11-28 10:15:00 | 745.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-11-27 13:00:00 | 758.00 | 2024-11-28 10:15:00 | 745.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-11-28 09:15:00 | 756.25 | 2024-11-28 10:15:00 | 745.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-03 10:15:00 | 766.35 | 2024-12-09 09:15:00 | 760.35 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-12-03 10:45:00 | 764.50 | 2024-12-09 09:15:00 | 760.35 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-27 15:00:00 | 686.40 | 2025-01-01 11:15:00 | 690.70 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-12-30 10:45:00 | 686.60 | 2025-01-01 11:15:00 | 690.70 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-01-02 11:15:00 | 691.00 | 2025-01-06 09:15:00 | 681.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-02 12:45:00 | 690.85 | 2025-01-06 09:15:00 | 681.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest1 | 2025-01-08 11:45:00 | 663.65 | 2025-01-09 09:15:00 | 678.25 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-01-09 15:15:00 | 667.70 | 2025-01-22 11:15:00 | 634.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 667.70 | 2025-01-22 14:15:00 | 645.50 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-01-13 09:15:00 | 664.80 | 2025-01-23 11:15:00 | 652.90 | STOP_HIT | 1.00 | 1.79% |
| SELL | retest2 | 2025-02-03 09:15:00 | 619.40 | 2025-02-04 13:15:00 | 633.15 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-02-04 11:30:00 | 620.65 | 2025-02-04 13:15:00 | 633.15 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-02-18 12:45:00 | 578.15 | 2025-02-24 12:15:00 | 583.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-02-19 09:15:00 | 578.85 | 2025-02-24 12:15:00 | 583.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-02-19 10:45:00 | 577.35 | 2025-02-24 12:15:00 | 583.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-02-19 12:00:00 | 578.15 | 2025-02-24 12:15:00 | 583.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-02-21 11:15:00 | 574.30 | 2025-02-24 12:15:00 | 583.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-02-24 09:15:00 | 569.20 | 2025-02-24 12:15:00 | 583.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-03-04 11:15:00 | 558.50 | 2025-03-05 14:15:00 | 570.50 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-05 09:15:00 | 551.50 | 2025-03-05 14:15:00 | 570.50 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-03-05 09:45:00 | 557.40 | 2025-03-05 14:15:00 | 570.50 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-03-07 10:15:00 | 571.05 | 2025-03-10 10:15:00 | 562.15 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-03-10 09:30:00 | 566.30 | 2025-03-10 10:15:00 | 562.15 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-03-12 15:00:00 | 572.80 | 2025-03-17 10:15:00 | 569.75 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-03-17 09:15:00 | 572.40 | 2025-03-17 10:15:00 | 569.75 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-03-17 10:00:00 | 571.50 | 2025-03-17 10:15:00 | 569.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-04-04 10:15:00 | 631.20 | 2025-04-07 09:15:00 | 568.08 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 12:00:00 | 631.00 | 2025-04-07 09:15:00 | 567.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 13:00:00 | 631.05 | 2025-04-07 09:15:00 | 567.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 13:30:00 | 629.55 | 2025-04-07 09:15:00 | 566.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-24 11:15:00 | 656.60 | 2025-04-30 13:15:00 | 623.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:30:00 | 649.50 | 2025-04-30 15:15:00 | 617.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 11:15:00 | 656.60 | 2025-05-02 14:15:00 | 629.60 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-04-25 09:30:00 | 649.50 | 2025-05-02 14:15:00 | 629.60 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2025-05-26 12:15:00 | 631.95 | 2025-05-28 11:15:00 | 637.05 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-27 12:15:00 | 634.25 | 2025-05-28 11:15:00 | 637.05 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-05-27 13:00:00 | 632.65 | 2025-05-28 11:15:00 | 637.05 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-02 13:15:00 | 633.00 | 2025-06-02 14:15:00 | 637.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-05 10:30:00 | 628.00 | 2025-06-09 14:15:00 | 626.55 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-06-17 11:15:00 | 624.50 | 2025-06-17 11:15:00 | 629.85 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-23 09:15:00 | 600.85 | 2025-06-27 12:15:00 | 598.95 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-06-23 10:30:00 | 605.10 | 2025-06-27 12:15:00 | 598.95 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-06-30 14:30:00 | 599.40 | 2025-07-01 09:15:00 | 595.25 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-07 09:15:00 | 584.75 | 2025-07-14 14:15:00 | 573.40 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2025-07-21 12:00:00 | 579.20 | 2025-07-25 09:15:00 | 582.40 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-07-30 15:00:00 | 568.35 | 2025-08-11 12:15:00 | 567.55 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-08 10:45:00 | 552.85 | 2025-09-09 15:15:00 | 564.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-09-09 10:00:00 | 554.80 | 2025-09-09 15:15:00 | 564.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-09 12:00:00 | 554.80 | 2025-09-09 15:15:00 | 564.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-30 09:45:00 | 522.00 | 2025-10-01 09:15:00 | 545.30 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-09-30 13:00:00 | 522.00 | 2025-10-01 09:15:00 | 545.30 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-10-17 10:30:00 | 563.15 | 2025-10-23 10:15:00 | 574.45 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest1 | 2025-10-30 15:15:00 | 572.30 | 2025-10-31 09:15:00 | 563.80 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-11 09:45:00 | 549.40 | 2025-11-11 12:15:00 | 557.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-13 13:45:00 | 564.75 | 2025-11-14 10:15:00 | 559.55 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-13 14:15:00 | 564.65 | 2025-11-14 10:15:00 | 559.55 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-17 09:15:00 | 572.00 | 2025-11-18 12:15:00 | 560.80 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-11-28 15:15:00 | 557.00 | 2025-12-02 09:15:00 | 553.15 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-12-01 10:30:00 | 557.05 | 2025-12-02 09:15:00 | 553.15 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-03 10:45:00 | 552.25 | 2025-12-03 12:15:00 | 555.70 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-05 09:15:00 | 549.90 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-08 09:30:00 | 550.25 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-12-08 10:00:00 | 550.95 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-08 12:00:00 | 550.85 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-31 09:30:00 | 580.40 | 2026-01-05 10:15:00 | 582.70 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2026-01-09 09:15:00 | 556.10 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-01-09 10:00:00 | 558.10 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2026-01-09 11:45:00 | 557.90 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2026-01-13 09:30:00 | 557.15 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2026-01-13 14:15:00 | 555.65 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2026-01-13 15:15:00 | 554.00 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2026-01-14 10:30:00 | 555.60 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2026-01-14 14:15:00 | 555.40 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-01-19 09:15:00 | 537.00 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-29 09:15:00 | 537.90 | 2026-01-30 10:15:00 | 542.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-29 14:30:00 | 538.80 | 2026-01-30 10:15:00 | 542.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-30 09:15:00 | 532.45 | 2026-01-30 10:15:00 | 542.10 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-23 10:45:00 | 581.95 | 2026-02-23 13:15:00 | 586.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-23 11:15:00 | 581.95 | 2026-02-23 13:15:00 | 586.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-23 11:45:00 | 581.55 | 2026-02-23 13:15:00 | 586.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-03-06 15:15:00 | 596.10 | 2026-03-12 09:15:00 | 566.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 596.10 | 2026-03-12 10:15:00 | 582.30 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2026-03-17 12:15:00 | 565.90 | 2026-03-18 09:15:00 | 582.50 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2026-03-17 13:45:00 | 566.65 | 2026-03-18 09:15:00 | 582.50 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-03-17 15:00:00 | 565.95 | 2026-03-18 09:15:00 | 582.50 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-02 09:15:00 | 566.55 | 2026-04-02 12:15:00 | 584.10 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-04-09 11:30:00 | 597.40 | 2026-04-09 15:15:00 | 589.15 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-21 09:15:00 | 651.00 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-21 11:45:00 | 650.60 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-22 10:45:00 | 650.45 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-04-22 12:45:00 | 655.40 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-04-29 13:00:00 | 593.75 | 2026-04-30 14:15:00 | 605.75 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-30 10:00:00 | 593.00 | 2026-04-30 14:15:00 | 605.75 | STOP_HIT | 1.00 | -2.15% |
