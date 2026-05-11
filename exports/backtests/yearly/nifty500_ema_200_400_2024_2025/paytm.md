# One 97 Communications Ltd. (PAYTM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1188.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 0 |
| TARGET_HIT | 10 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 10 / 12 / 0
- **Avg / median % per leg:** 2.62% / 1.08%
- **Sum % (uncompounded):** 57.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 11 | 91.7% | 10 | 2 | 0 | 8.16% | 97.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 11 | 91.7% | 10 | 2 | 0 | 8.16% | 97.9% |
| SELL (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -4.03% | -40.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -4.03% | -40.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 11 | 50.0% | 10 | 12 | 0 | 2.62% | 57.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 468.30 | 399.70 | 399.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 10:15:00 | 471.25 | 400.41 | 399.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 426.70 | 427.16 | 415.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 426.70 | 427.16 | 415.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 865.50 | 949.20 | 887.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 865.50 | 949.20 | 887.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 869.55 | 948.41 | 887.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 860.05 | 948.41 | 887.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 909.15 | 917.09 | 881.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:30:00 | 905.05 | 917.09 | 881.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 848.90 | 915.58 | 881.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 848.90 | 915.58 | 881.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 832.70 | 914.76 | 881.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 832.70 | 914.76 | 881.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 761.00 | 858.27 | 858.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 15:15:00 | 741.75 | 856.00 | 857.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 14:15:00 | 743.55 | 735.74 | 772.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 15:00:00 | 743.55 | 735.74 | 772.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 777.45 | 738.98 | 769.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:45:00 | 776.55 | 738.98 | 769.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 773.05 | 739.32 | 769.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 15:00:00 | 764.45 | 739.57 | 769.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:30:00 | 762.05 | 740.75 | 769.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 780.85 | 741.15 | 769.79 | SL hit (close>static) qty=1.00 sl=779.70 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 866.05 | 785.94 | 785.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 870.50 | 792.61 | 789.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 830.10 | 830.84 | 812.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 12:45:00 | 831.40 | 830.84 | 812.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 816.50 | 832.62 | 814.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 863.90 | 832.62 | 814.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 861.65 | 832.91 | 814.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:30:00 | 874.10 | 833.36 | 815.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 870.00 | 834.95 | 816.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 15:00:00 | 872.60 | 841.96 | 824.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:45:00 | 871.90 | 843.23 | 825.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 825.55 | 843.34 | 827.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 824.65 | 843.34 | 827.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 823.10 | 843.14 | 827.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 823.10 | 843.14 | 827.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 832.50 | 842.65 | 827.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 829.60 | 842.65 | 827.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-06-05 09:15:00 | 961.51 | 867.31 | 844.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 1171.80 | 1279.94 | 1280.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 1160.00 | 1272.57 | 1276.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1201.50 | 1200.51 | 1229.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 15:00:00 | 1201.50 | 1200.51 | 1229.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 1104.45 | 1052.33 | 1105.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:45:00 | 1105.80 | 1052.33 | 1105.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 1101.65 | 1052.82 | 1105.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 1106.70 | 1052.82 | 1105.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 1112.00 | 1053.41 | 1105.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 1112.00 | 1053.41 | 1105.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 1116.00 | 1054.03 | 1105.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 1119.15 | 1054.03 | 1105.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1123.25 | 1055.32 | 1105.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:00:00 | 1123.25 | 1055.32 | 1105.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1103.60 | 1058.34 | 1105.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1061.60 | 1103.82 | 1118.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 1120.25 | 1104.09 | 1118.12 | SL hit (close>static) qty=1.00 sl=1115.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-24 15:00:00 | 764.45 | 2025-03-25 12:15:00 | 780.85 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-25 11:30:00 | 762.05 | 2025-03-25 12:15:00 | 780.85 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-05-07 10:30:00 | 874.10 | 2025-06-05 09:15:00 | 961.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 14:30:00 | 870.00 | 2025-06-05 09:15:00 | 957.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-19 15:00:00 | 872.60 | 2025-06-05 09:15:00 | 959.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-20 11:45:00 | 871.90 | 2025-06-05 09:15:00 | 959.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-18 09:30:00 | 875.10 | 2025-07-14 09:15:00 | 962.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-19 13:30:00 | 872.30 | 2025-07-14 09:15:00 | 959.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 12:45:00 | 871.00 | 2025-07-14 09:15:00 | 958.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 15:00:00 | 873.55 | 2025-07-14 09:15:00 | 960.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 09:45:00 | 1156.00 | 2025-10-15 09:15:00 | 1271.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 10:45:00 | 1162.60 | 2025-10-15 09:15:00 | 1278.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-23 14:15:00 | 1161.60 | 2026-01-27 12:15:00 | 1124.60 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2026-01-28 09:15:00 | 1159.30 | 2026-01-28 11:15:00 | 1171.80 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2026-04-27 09:15:00 | 1061.60 | 2026-04-27 11:15:00 | 1120.25 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1090.20 | 2026-05-04 09:15:00 | 1117.10 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-04-30 10:00:00 | 1090.85 | 2026-05-04 09:15:00 | 1117.10 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1092.50 | 2026-05-04 09:15:00 | 1117.10 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-05-04 12:15:00 | 1110.70 | 2026-05-07 09:15:00 | 1173.20 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2026-05-04 14:30:00 | 1108.20 | 2026-05-07 09:15:00 | 1173.20 | STOP_HIT | 1.00 | -5.87% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1107.00 | 2026-05-07 09:15:00 | 1173.20 | STOP_HIT | 1.00 | -5.98% |
| SELL | retest2 | 2026-05-05 10:15:00 | 1111.30 | 2026-05-07 09:15:00 | 1173.20 | STOP_HIT | 1.00 | -5.57% |
