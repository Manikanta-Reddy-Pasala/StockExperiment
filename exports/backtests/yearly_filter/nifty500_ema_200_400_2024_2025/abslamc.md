# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1075.00
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
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 22
- **Target hits / Stop hits / Partials:** 5 / 22 / 0
- **Avg / median % per leg:** -0.46% / -2.73%
- **Sum % (uncompounded):** -12.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 5 | 7 | 0 | 2.75% | 32.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 5 | 41.7% | 5 | 7 | 0 | 2.75% | 32.9% |
| SELL (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.03% | -45.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.03% | -45.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 5 | 18.5% | 5 | 22 | 0 | -0.46% | -12.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 14:15:00 | 755.75 | 795.21 | 795.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 739.20 | 793.81 | 794.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 636.75 | 635.39 | 673.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:30:00 | 636.10 | 635.39 | 673.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 664.10 | 639.62 | 665.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:45:00 | 664.60 | 639.62 | 665.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 659.50 | 636.05 | 656.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:45:00 | 656.25 | 636.05 | 656.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 656.80 | 636.26 | 656.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:30:00 | 657.80 | 636.26 | 656.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 655.25 | 636.45 | 656.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:45:00 | 654.50 | 636.45 | 656.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 643.25 | 636.52 | 656.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 10:15:00 | 641.35 | 641.69 | 657.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 12:15:00 | 642.35 | 641.66 | 657.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 13:15:00 | 640.50 | 641.66 | 656.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 640.00 | 641.71 | 656.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 642.20 | 641.54 | 656.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:45:00 | 644.50 | 641.54 | 656.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 660.10 | 641.75 | 656.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 660.10 | 641.75 | 656.17 | SL hit (close>static) qty=1.00 sl=656.65 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 716.80 | 662.71 | 662.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 722.00 | 672.83 | 667.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 736.25 | 741.50 | 714.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 14:45:00 | 734.45 | 741.50 | 714.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 833.75 | 855.00 | 828.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 833.75 | 855.00 | 828.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 830.70 | 853.50 | 829.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 830.70 | 853.50 | 829.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 827.60 | 853.24 | 829.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 827.60 | 853.24 | 829.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 823.10 | 852.94 | 829.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:30:00 | 834.45 | 852.78 | 829.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 831.90 | 852.18 | 829.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 829.80 | 851.94 | 829.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 831.00 | 850.23 | 829.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 835.75 | 849.90 | 829.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 848.25 | 848.84 | 829.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:45:00 | 850.00 | 849.03 | 830.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 846.55 | 850.59 | 834.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 822.50 | 849.02 | 834.17 | SL hit (close<static) qty=1.00 sl=825.25 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 809.70 | 824.63 | 824.69 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 854.25 | 824.62 | 824.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 11:15:00 | 860.90 | 824.98 | 824.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 830.45 | 837.53 | 831.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 830.45 | 837.53 | 831.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 830.45 | 837.53 | 831.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 830.45 | 837.53 | 831.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 831.95 | 837.48 | 831.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 834.75 | 837.48 | 831.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 830.45 | 837.41 | 831.79 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 773.15 | 826.57 | 826.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 771.10 | 826.02 | 826.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 753.70 | 745.27 | 768.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:45:00 | 753.15 | 745.27 | 768.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 765.40 | 746.62 | 767.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 765.40 | 746.62 | 767.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 771.80 | 746.87 | 767.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 771.80 | 746.87 | 767.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 769.00 | 747.09 | 767.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 769.15 | 747.09 | 767.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 768.85 | 747.30 | 767.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 765.10 | 747.48 | 767.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 788.25 | 748.85 | 767.37 | SL hit (close>static) qty=1.00 sl=775.35 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 855.00 | 777.40 | 777.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 862.00 | 797.54 | 791.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 859.40 | 859.99 | 832.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 09:30:00 | 858.50 | 859.99 | 832.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 876.90 | 911.02 | 877.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 876.90 | 911.02 | 877.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 872.00 | 910.63 | 877.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:45:00 | 870.90 | 910.63 | 877.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 883.45 | 910.36 | 877.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 909.80 | 909.35 | 877.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 889.00 | 907.61 | 878.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 977.90 | 912.47 | 883.64 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-03 09:15:00 | 541.10 | 2024-06-06 09:15:00 | 579.21 | TARGET_HIT | 1.00 | 7.04% |
| BUY | retest2 | 2024-06-04 13:30:00 | 526.55 | 2024-06-06 09:15:00 | 577.50 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2024-06-04 15:15:00 | 525.00 | 2024-06-10 09:15:00 | 595.21 | TARGET_HIT | 1.00 | 13.37% |
| SELL | retest2 | 2025-04-25 10:15:00 | 641.35 | 2025-04-29 09:15:00 | 660.10 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-04-25 12:15:00 | 642.35 | 2025-04-29 09:15:00 | 660.10 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-04-25 13:15:00 | 640.50 | 2025-04-29 09:15:00 | 660.10 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-04-25 15:15:00 | 640.00 | 2025-04-29 09:15:00 | 660.10 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-05-02 11:45:00 | 648.50 | 2025-05-12 09:15:00 | 671.70 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-05-02 12:30:00 | 647.95 | 2025-05-12 09:15:00 | 671.70 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2025-05-05 09:30:00 | 646.40 | 2025-05-12 09:15:00 | 671.70 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-05-06 11:15:00 | 645.85 | 2025-05-12 09:15:00 | 671.70 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-05-08 15:00:00 | 644.15 | 2025-05-12 09:15:00 | 671.70 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-09-03 09:30:00 | 834.45 | 2025-09-17 13:15:00 | 822.50 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-03 13:00:00 | 831.90 | 2025-09-17 13:15:00 | 822.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-03 14:15:00 | 829.80 | 2025-09-17 13:15:00 | 822.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-04 15:15:00 | 831.00 | 2025-09-17 14:15:00 | 814.85 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-09-08 09:45:00 | 848.25 | 2025-09-17 14:15:00 | 814.85 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2025-09-09 10:45:00 | 850.00 | 2025-09-17 14:15:00 | 814.85 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2025-09-16 09:15:00 | 846.55 | 2025-09-17 14:15:00 | 814.85 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-12-17 10:30:00 | 765.10 | 2025-12-18 09:15:00 | 788.25 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-12-19 10:00:00 | 765.80 | 2025-12-23 15:15:00 | 779.80 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-22 10:15:00 | 764.45 | 2025-12-23 15:15:00 | 779.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-12-22 12:30:00 | 764.75 | 2025-12-23 15:15:00 | 779.80 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-12-24 11:45:00 | 772.80 | 2025-12-29 13:15:00 | 793.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-12-26 09:15:00 | 771.95 | 2025-12-29 13:15:00 | 793.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2026-04-01 09:15:00 | 909.80 | 2026-04-08 09:15:00 | 977.90 | TARGET_HIT | 1.00 | 7.49% |
| BUY | retest2 | 2026-04-02 13:45:00 | 889.00 | 2026-04-10 12:15:00 | 1000.78 | TARGET_HIT | 1.00 | 12.57% |
