# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1075.00
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
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 0
- **Avg / median % per leg:** -0.75% / -1.97%
- **Sum % (uncompounded):** -11.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 2 | 7 | 0 | 0.32% | 2.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 2 | 7 | 0 | 0.32% | 2.9% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.36% | -14.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.36% | -14.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 2 | 13.3% | 2 | 13 | 0 | -0.75% | -11.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 716.80 | 662.71 | 662.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 722.00 | 672.83 | 667.86 | Break + close above crossover candle high |
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
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 822.50 | 849.02 | 834.17 | SL hit (close<static) qty=1.00 sl=825.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 822.50 | 849.02 | 834.17 | SL hit (close<static) qty=1.00 sl=825.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 814.85 | 848.68 | 834.08 | SL hit (close<static) qty=1.00 sl=821.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 814.85 | 848.68 | 834.08 | SL hit (close<static) qty=1.00 sl=821.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 814.85 | 848.68 | 834.08 | SL hit (close<static) qty=1.00 sl=821.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 814.85 | 848.68 | 834.08 | SL hit (close<static) qty=1.00 sl=821.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 809.70 | 824.63 | 824.69 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-10-14 10:15:00)

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

### Cycle 4 — SELL (started 2025-10-31 09:15:00)

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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:00:00 | 765.80 | 750.98 | 767.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 764.45 | 751.62 | 767.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 764.75 | 751.97 | 767.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 761.80 | 752.12 | 767.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:30:00 | 763.65 | 752.12 | 767.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 762.00 | 752.22 | 767.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 765.10 | 752.35 | 767.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 774.70 | 752.99 | 767.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 774.70 | 752.99 | 767.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 779.80 | 753.26 | 767.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 779.80 | 753.26 | 767.40 | SL hit (close>static) qty=1.00 sl=775.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 779.80 | 753.26 | 767.40 | SL hit (close>static) qty=1.00 sl=775.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 779.80 | 753.26 | 767.40 | SL hit (close>static) qty=1.00 sl=775.35 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 772.80 | 753.94 | 767.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 771.95 | 754.75 | 767.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 793.00 | 756.70 | 767.93 | SL hit (close>static) qty=1.00 sl=782.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 793.00 | 756.70 | 767.93 | SL hit (close>static) qty=1.00 sl=782.95 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-05 15:15:00)

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
| Target hit | 2026-04-10 12:15:00 | 1000.78 | 922.56 | 891.24 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
