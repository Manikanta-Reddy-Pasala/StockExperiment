# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 808.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 11 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 14
- **Target hits / Stop hits / Partials:** 0 / 14 / 4
- **Avg / median % per leg:** -1.75% / -1.98%
- **Sum % (uncompounded):** -31.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.42% | -1.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.42% | -1.4% |
| SELL (all) | 17 | 4 | 23.5% | 0 | 13 | 4 | -1.77% | -30.1% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 3 | 2 | -2.99% | -15.0% |
| SELL @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 0 | 10 | 2 | -1.26% | -15.1% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 3 | 2 | -2.99% | -15.0% |
| retest2 (combined) | 13 | 3 | 23.1% | 0 | 11 | 2 | -1.27% | -16.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 11:15:00 | 882.15 | 1001.41 | 1001.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 854.15 | 940.31 | 964.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 947.20 | 935.45 | 960.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 947.20 | 935.45 | 960.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 947.20 | 935.45 | 960.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 947.20 | 935.45 | 960.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 953.05 | 935.11 | 959.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:45:00 | 953.45 | 935.11 | 959.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 957.90 | 935.52 | 959.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 13:30:00 | 964.40 | 935.52 | 959.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 14:15:00 | 954.75 | 935.72 | 959.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 14:30:00 | 961.35 | 935.72 | 959.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 994.10 | 936.45 | 959.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:30:00 | 996.65 | 936.45 | 959.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 1003.20 | 937.11 | 959.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 1005.90 | 937.11 | 959.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 14:15:00 | 1055.75 | 977.78 | 977.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 11:15:00 | 1060.85 | 988.54 | 983.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 10:15:00 | 1052.00 | 1056.29 | 1030.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 1052.00 | 1056.29 | 1030.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1027.10 | 1055.85 | 1030.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 1027.10 | 1055.85 | 1030.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1017.90 | 1055.47 | 1030.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 1011.50 | 1055.47 | 1030.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 995.85 | 1054.14 | 1029.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:45:00 | 992.80 | 1054.14 | 1029.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 15:15:00 | 902.00 | 1012.84 | 1013.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 888.95 | 982.99 | 988.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 764.45 | 762.72 | 821.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 764.45 | 762.72 | 821.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 804.30 | 769.70 | 814.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:30:00 | 803.75 | 770.49 | 814.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 13:15:00 | 819.80 | 770.99 | 814.61 | SL hit (close>static) qty=1.00 sl=815.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 1008.00 | 839.16 | 838.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 14:15:00 | 1019.35 | 840.95 | 839.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 890.90 | 895.52 | 875.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 09:45:00 | 891.60 | 895.52 | 875.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 882.85 | 897.15 | 878.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:00:00 | 882.85 | 897.15 | 878.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 876.15 | 896.94 | 878.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 876.15 | 896.94 | 878.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 870.35 | 896.67 | 878.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 870.35 | 896.67 | 878.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 874.05 | 896.45 | 878.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:30:00 | 879.05 | 896.15 | 878.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 10:15:00 | 866.55 | 896.72 | 880.01 | SL hit (close<static) qty=1.00 sl=870.05 alert=retest2 |

### Cycle 5 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 798.55 | 866.85 | 867.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 797.90 | 855.24 | 860.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 12:15:00 | 706.85 | 706.05 | 752.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:30:00 | 708.00 | 706.05 | 752.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 719.35 | 686.89 | 724.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 12:00:00 | 719.35 | 686.89 | 724.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 13:15:00 | 727.85 | 687.63 | 724.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 13:45:00 | 731.00 | 687.63 | 724.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 754.30 | 688.29 | 724.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 15:00:00 | 754.30 | 688.29 | 724.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 12:15:00 | 795.60 | 746.72 | 746.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 09:15:00 | 799.60 | 748.35 | 747.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1078.30 | 1106.49 | 1005.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1078.30 | 1106.49 | 1005.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1091.70 | 1149.12 | 1091.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 1091.70 | 1149.12 | 1091.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1019.40 | 1147.83 | 1090.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1019.40 | 1147.83 | 1090.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1033.60 | 1146.70 | 1090.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:45:00 | 1032.10 | 1146.70 | 1090.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 933.50 | 1055.30 | 1055.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 926.90 | 1049.63 | 1052.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 993.40 | 980.32 | 1006.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 993.40 | 980.32 | 1006.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1007.35 | 981.01 | 1006.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1007.35 | 981.01 | 1006.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1010.25 | 981.30 | 1006.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 1013.05 | 981.30 | 1006.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1017.80 | 981.67 | 1006.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 1017.80 | 981.67 | 1006.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1018.00 | 982.03 | 1006.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1011.35 | 982.03 | 1006.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:30:00 | 1010.20 | 982.58 | 1006.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1008.40 | 983.93 | 1006.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1031.35 | 984.41 | 1006.68 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1206.40 | 1007.66 | 1007.62 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 1007.50 | 1050.16 | 1050.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 1003.00 | 1049.25 | 1049.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 682.45 | 676.55 | 741.50 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:00:00 | 673.75 | 676.96 | 738.55 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 644.10 | 677.37 | 735.75 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 09:15:00 | 640.06 | 677.06 | 735.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:45:00 | 673.00 | 675.56 | 731.97 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 13:15:00 | 681.85 | 675.59 | 731.42 | SL hit (close>ema200) qty=0.50 sl=675.59 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-10 12:30:00 | 803.75 | 2024-12-10 13:15:00 | 819.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-01-22 14:30:00 | 879.05 | 2025-01-27 10:15:00 | 866.55 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-11 09:15:00 | 1011.35 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-09-11 10:30:00 | 1010.20 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-09-12 09:15:00 | 1008.40 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-15 09:15:00 | 1013.00 | 2025-09-15 09:15:00 | 1020.85 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-13 09:15:00 | 993.95 | 2025-10-14 12:15:00 | 944.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 993.95 | 2025-10-15 09:15:00 | 997.05 | STOP_HIT | 0.50 | -0.31% |
| SELL | retest2 | 2025-10-15 12:00:00 | 1000.15 | 2025-10-20 09:15:00 | 950.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 12:00:00 | 1000.15 | 2025-10-24 13:15:00 | 992.55 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2026-04-09 13:00:00 | 673.75 | 2026-04-13 09:15:00 | 640.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-09 13:00:00 | 673.75 | 2026-04-15 13:15:00 | 681.85 | STOP_HIT | 0.50 | -1.20% |
| SELL | retest1 | 2026-04-13 09:15:00 | 644.10 | 2026-04-22 09:15:00 | 664.24 | PARTIAL | 0.50 | -3.13% |
| SELL | retest1 | 2026-04-13 09:15:00 | 644.10 | 2026-04-22 13:15:00 | 695.00 | STOP_HIT | 0.50 | -7.90% |
| SELL | retest1 | 2026-04-15 11:45:00 | 673.00 | 2026-04-28 09:15:00 | 725.05 | STOP_HIT | 1.00 | -7.73% |
| SELL | retest2 | 2026-04-16 11:30:00 | 699.20 | 2026-04-28 09:15:00 | 725.05 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-24 09:15:00 | 695.30 | 2026-04-28 10:15:00 | 739.55 | STOP_HIT | 1.00 | -6.36% |
| SELL | retest2 | 2026-04-24 10:15:00 | 695.00 | 2026-04-28 10:15:00 | 739.55 | STOP_HIT | 1.00 | -6.41% |
