# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 835.00
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
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 14 |
| TARGET_HIT | 10 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 26
- **Target hits / Stop hits / Partials:** 9 / 30 / 14
- **Avg / median % per leg:** 1.95% / 0.25%
- **Sum % (uncompounded):** 103.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.03% | -6.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.03% | -6.1% |
| SELL (all) | 51 | 27 | 52.9% | 9 | 28 | 14 | 2.15% | 109.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 51 | 27 | 52.9% | 9 | 28 | 14 | 2.15% | 109.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 27 | 50.9% | 9 | 30 | 14 | 1.95% | 103.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 884.60 | 949.80 | 950.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 865.98 | 945.85 | 948.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 699.06 | 695.37 | 756.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 10:00:00 | 699.06 | 695.37 | 756.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 753.39 | 697.93 | 752.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 11:45:00 | 753.62 | 697.93 | 752.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 762.56 | 698.57 | 752.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:45:00 | 763.80 | 698.57 | 752.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 761.47 | 699.19 | 753.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 757.00 | 700.44 | 753.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 764.98 | 703.77 | 752.75 | SL hit (close>static) qty=1.00 sl=763.99 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 787.20 | 758.59 | 758.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 791.52 | 758.91 | 758.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 794.42 | 803.90 | 785.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 794.42 | 803.90 | 785.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 794.42 | 803.90 | 785.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:00:00 | 807.12 | 803.15 | 786.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 14:00:00 | 804.82 | 807.63 | 790.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 781.54 | 832.42 | 819.53 | SL hit (close<static) qty=1.00 sl=785.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 750.46 | 808.67 | 808.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 743.04 | 796.23 | 802.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 772.68 | 772.09 | 783.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:30:00 | 774.38 | 772.09 | 783.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 780.82 | 772.47 | 782.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 776.70 | 772.70 | 782.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 791.42 | 773.01 | 781.78 | SL hit (close>static) qty=1.00 sl=784.20 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 805.00 | 778.07 | 778.03 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 751.50 | 778.44 | 778.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 746.00 | 777.38 | 778.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 767.70 | 764.28 | 770.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 767.70 | 764.28 | 770.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 771.10 | 764.09 | 769.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 772.70 | 764.09 | 769.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 771.50 | 764.17 | 769.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 773.80 | 764.17 | 769.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 757.40 | 756.29 | 763.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:15:00 | 744.90 | 756.04 | 763.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 745.00 | 755.84 | 763.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 707.65 | 752.81 | 761.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 707.75 | 752.81 | 761.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 12:15:00 | 670.41 | 724.86 | 741.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 774.20 | 699.61 | 699.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 779.45 | 720.36 | 711.15 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-20 09:15:00 | 757.00 | 2025-03-21 09:15:00 | 764.98 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-21 09:45:00 | 757.90 | 2025-03-21 10:15:00 | 767.61 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-03-21 13:30:00 | 757.00 | 2025-03-21 14:15:00 | 764.30 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-03-25 09:15:00 | 748.01 | 2025-04-03 12:15:00 | 758.73 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-03-28 15:15:00 | 742.40 | 2025-04-03 12:15:00 | 758.73 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-04-03 09:45:00 | 744.74 | 2025-04-03 14:15:00 | 765.69 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-04-04 10:00:00 | 744.10 | 2025-04-07 09:15:00 | 706.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 11:45:00 | 742.53 | 2025-04-07 09:15:00 | 705.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 10:00:00 | 744.10 | 2025-04-08 09:15:00 | 728.00 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2025-04-04 11:45:00 | 742.53 | 2025-04-08 09:15:00 | 728.00 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2025-05-06 09:15:00 | 744.66 | 2025-05-09 09:15:00 | 707.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 10:15:00 | 746.52 | 2025-05-09 09:15:00 | 709.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 744.66 | 2025-05-16 09:15:00 | 765.22 | STOP_HIT | 0.50 | -2.76% |
| SELL | retest2 | 2025-05-08 10:15:00 | 746.52 | 2025-05-16 09:15:00 | 765.22 | STOP_HIT | 0.50 | -2.50% |
| BUY | retest2 | 2025-06-16 13:00:00 | 807.12 | 2025-07-28 12:15:00 | 781.54 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-06-19 14:00:00 | 804.82 | 2025-07-28 12:15:00 | 781.54 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-09-11 12:45:00 | 776.70 | 2025-09-17 11:15:00 | 791.42 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-09-24 14:45:00 | 776.24 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-25 14:30:00 | 776.18 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-10-07 10:00:00 | 776.40 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-14 11:15:00 | 765.36 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-10-16 14:45:00 | 765.56 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-10-17 10:00:00 | 764.40 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-10-17 13:30:00 | 764.08 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-10-24 13:30:00 | 776.44 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-29 09:15:00 | 761.36 | 2025-10-30 09:15:00 | 787.62 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-11-04 09:15:00 | 773.96 | 2025-11-07 09:15:00 | 735.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 773.96 | 2025-11-12 09:15:00 | 796.06 | STOP_HIT | 0.50 | -2.86% |
| SELL | retest2 | 2026-01-08 13:15:00 | 744.90 | 2026-01-12 11:15:00 | 707.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 745.00 | 2026-01-12 11:15:00 | 707.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 744.90 | 2026-02-01 12:15:00 | 670.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 745.00 | 2026-02-01 12:15:00 | 670.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 10:45:00 | 745.45 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-02-11 14:15:00 | 746.60 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2026-02-12 11:15:00 | 733.80 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-12 13:00:00 | 733.20 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-12 13:45:00 | 733.80 | 2026-02-24 11:15:00 | 709.27 | PARTIAL | 0.50 | 3.34% |
| SELL | retest2 | 2026-02-16 14:00:00 | 733.70 | 2026-02-24 12:15:00 | 708.18 | PARTIAL | 0.50 | 3.48% |
| SELL | retest2 | 2026-02-17 12:00:00 | 732.45 | 2026-02-27 09:15:00 | 695.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 732.30 | 2026-02-27 09:15:00 | 695.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 731.50 | 2026-02-27 09:15:00 | 694.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 730.95 | 2026-02-27 09:15:00 | 694.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 730.00 | 2026-02-27 09:15:00 | 693.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:45:00 | 733.80 | 2026-03-02 09:15:00 | 670.91 | TARGET_HIT | 0.50 | 8.57% |
| SELL | retest2 | 2026-02-16 14:00:00 | 733.70 | 2026-03-02 09:15:00 | 671.94 | TARGET_HIT | 0.50 | 8.42% |
| SELL | retest2 | 2026-02-17 12:00:00 | 732.45 | 2026-03-02 09:15:00 | 659.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 732.30 | 2026-03-02 09:15:00 | 659.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 731.50 | 2026-03-02 09:15:00 | 658.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 730.95 | 2026-03-02 09:15:00 | 657.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 730.00 | 2026-03-02 09:15:00 | 657.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 09:30:00 | 731.10 | 2026-04-16 14:15:00 | 738.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-04-16 11:30:00 | 731.60 | 2026-04-16 14:15:00 | 738.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-04-17 09:30:00 | 731.50 | 2026-04-17 10:15:00 | 744.10 | STOP_HIT | 1.00 | -1.72% |
