# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1179.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 5 |
| TARGET_HIT | 11 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 46
- **Target hits / Stop hits / Partials:** 11 / 48 / 5
- **Avg / median % per leg:** 0.01% / -1.86%
- **Sum % (uncompounded):** 0.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 8 | 25.8% | 8 | 23 | 0 | 0.67% | 20.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 8 | 25.8% | 8 | 23 | 0 | 0.67% | 20.7% |
| SELL (all) | 33 | 10 | 30.3% | 3 | 25 | 5 | -0.61% | -20.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 10 | 30.3% | 3 | 25 | 5 | -0.61% | -20.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 64 | 18 | 28.1% | 11 | 48 | 5 | 0.01% | 0.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 969.95 | 1049.50 | 1049.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 965.00 | 1048.66 | 1049.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 11:15:00 | 988.30 | 980.97 | 1004.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 12:00:00 | 988.30 | 980.97 | 1004.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 984.65 | 981.01 | 1004.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 993.00 | 981.01 | 1004.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 982.15 | 981.02 | 1004.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:45:00 | 988.50 | 981.02 | 1004.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 1005.40 | 981.26 | 1004.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 1005.40 | 981.26 | 1004.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 984.40 | 981.29 | 1004.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:15:00 | 959.00 | 981.29 | 1004.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 951.45 | 980.99 | 1004.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 944.10 | 980.06 | 1003.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 1044.50 | 983.56 | 1002.98 | SL hit (close>static) qty=1.00 sl=1042.95 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 1084.00 | 1017.92 | 1017.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 1091.85 | 1021.08 | 1019.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 12:15:00 | 1039.95 | 1040.18 | 1030.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 12:45:00 | 1041.45 | 1040.18 | 1030.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1033.70 | 1040.91 | 1031.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 1033.70 | 1040.91 | 1031.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1026.25 | 1040.76 | 1031.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 1026.25 | 1040.76 | 1031.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1019.65 | 1040.55 | 1031.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 1019.65 | 1040.55 | 1031.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1033.05 | 1040.03 | 1031.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:15:00 | 1040.05 | 1039.93 | 1031.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:30:00 | 1040.15 | 1039.69 | 1031.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:45:00 | 1037.90 | 1041.80 | 1033.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 1026.40 | 1041.65 | 1033.18 | SL hit (close<static) qty=1.00 sl=1028.85 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 916.55 | 1036.73 | 1036.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 910.65 | 1035.48 | 1036.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 995.00 | 994.42 | 1013.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 15:00:00 | 995.00 | 994.42 | 1013.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1015.30 | 992.41 | 1010.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1015.30 | 992.41 | 1010.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1017.30 | 992.66 | 1010.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 1018.40 | 992.66 | 1010.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1023.80 | 994.67 | 1011.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 1023.80 | 994.67 | 1011.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1020.50 | 995.07 | 1011.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:15:00 | 1021.45 | 995.07 | 1011.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1019.05 | 995.31 | 1011.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:15:00 | 1022.50 | 995.31 | 1011.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 994.65 | 1004.24 | 1014.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 12:30:00 | 1015.25 | 1004.24 | 1014.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1011.40 | 1003.48 | 1013.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 1015.95 | 1003.48 | 1013.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1003.15 | 1003.48 | 1013.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1010.50 | 1003.48 | 1013.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 937.60 | 894.21 | 935.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 932.85 | 894.21 | 935.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 939.45 | 894.66 | 935.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 939.45 | 894.66 | 935.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 933.60 | 895.05 | 935.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:15:00 | 930.60 | 895.05 | 935.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 941.90 | 895.84 | 935.71 | SL hit (close>static) qty=1.00 sl=940.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 992.30 | 917.77 | 917.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 1005.00 | 921.38 | 919.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1068.80 | 1069.13 | 1031.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 1068.80 | 1069.13 | 1031.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1049.20 | 1076.70 | 1049.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 1046.90 | 1076.70 | 1049.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1048.80 | 1076.43 | 1049.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 1048.00 | 1076.43 | 1049.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1061.90 | 1076.28 | 1049.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:30:00 | 1069.90 | 1076.17 | 1049.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 1074.00 | 1076.17 | 1049.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 12:15:00 | 1073.90 | 1075.85 | 1050.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 1069.60 | 1075.74 | 1050.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1047.80 | 1075.32 | 1050.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1047.80 | 1075.32 | 1050.69 | SL hit (close<static) qty=1.00 sl=1047.90 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1182.10 | 1260.87 | 1261.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1176.20 | 1260.03 | 1260.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1223.80 | 1202.80 | 1226.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1223.80 | 1202.80 | 1226.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1223.80 | 1202.80 | 1226.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:45:00 | 1209.40 | 1203.89 | 1225.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 1209.70 | 1203.96 | 1225.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1184.10 | 1204.42 | 1224.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 1205.80 | 1202.06 | 1222.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1211.90 | 1202.16 | 1222.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 1224.60 | 1202.16 | 1222.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1221.10 | 1202.35 | 1222.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 1220.70 | 1202.35 | 1222.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1223.70 | 1202.56 | 1222.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 1223.70 | 1202.56 | 1222.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1229.50 | 1202.83 | 1222.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1229.50 | 1202.83 | 1222.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1240.00 | 1204.35 | 1222.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 1234.00 | 1206.34 | 1223.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:00:00 | 1232.80 | 1206.60 | 1223.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 1250.30 | 1207.41 | 1223.31 | SL hit (close>static) qty=1.00 sl=1248.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-22 10:15:00 | 988.05 | 2024-07-30 09:15:00 | 1075.09 | TARGET_HIT | 1.00 | 8.81% |
| BUY | retest2 | 2024-07-22 11:15:00 | 981.45 | 2024-08-14 12:15:00 | 1079.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 15:00:00 | 977.35 | 2024-08-14 12:15:00 | 1079.49 | TARGET_HIT | 1.00 | 10.45% |
| BUY | retest2 | 2024-07-23 10:00:00 | 981.35 | 2024-08-14 13:15:00 | 1086.86 | TARGET_HIT | 1.00 | 10.75% |
| BUY | retest2 | 2024-08-08 09:15:00 | 1007.00 | 2024-08-14 13:15:00 | 1107.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-09 13:30:00 | 997.15 | 2024-08-14 13:15:00 | 1096.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-09 09:15:00 | 1006.80 | 2024-10-17 11:15:00 | 969.95 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2024-10-14 09:15:00 | 1000.00 | 2024-10-17 11:15:00 | 969.95 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-11-14 09:15:00 | 944.10 | 2024-11-19 12:15:00 | 1044.50 | STOP_HIT | 1.00 | -10.63% |
| BUY | retest2 | 2024-12-18 13:15:00 | 1040.05 | 2024-12-23 10:15:00 | 1026.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-12-19 11:30:00 | 1040.15 | 2024-12-23 10:15:00 | 1026.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-12-23 09:45:00 | 1037.90 | 2024-12-23 10:15:00 | 1026.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-12-23 13:00:00 | 1039.20 | 2024-12-23 14:15:00 | 1021.60 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-12-24 11:15:00 | 1030.45 | 2024-12-27 10:15:00 | 1024.10 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-12-24 12:30:00 | 1042.35 | 2025-01-21 09:15:00 | 1020.20 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-26 11:45:00 | 1032.85 | 2025-01-21 09:15:00 | 1020.20 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-12-26 12:15:00 | 1033.25 | 2025-01-21 09:15:00 | 1020.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-12-26 15:15:00 | 1052.05 | 2025-01-21 09:15:00 | 1020.20 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-12-30 13:30:00 | 1047.60 | 2025-01-21 09:15:00 | 1020.20 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-12-30 14:30:00 | 1041.90 | 2025-01-21 09:15:00 | 1020.20 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-12-31 13:15:00 | 1043.15 | 2025-01-21 09:15:00 | 1020.20 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-01-14 09:30:00 | 1055.25 | 2025-01-21 10:15:00 | 1004.45 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2025-01-14 15:00:00 | 1049.30 | 2025-01-21 10:15:00 | 1004.45 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2025-01-15 09:15:00 | 1066.60 | 2025-01-21 10:15:00 | 1004.45 | STOP_HIT | 1.00 | -5.83% |
| BUY | retest2 | 2025-01-16 09:15:00 | 1071.70 | 2025-01-21 10:15:00 | 1004.45 | STOP_HIT | 1.00 | -6.28% |
| SELL | retest2 | 2025-03-17 12:15:00 | 930.60 | 2025-03-17 13:15:00 | 941.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-03-17 15:15:00 | 931.00 | 2025-03-18 09:15:00 | 958.20 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-03-26 13:30:00 | 930.75 | 2025-03-28 14:15:00 | 884.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 14:15:00 | 931.15 | 2025-03-28 14:15:00 | 884.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 13:30:00 | 930.75 | 2025-04-04 10:15:00 | 837.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-26 14:15:00 | 931.15 | 2025-04-04 10:15:00 | 838.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-23 10:15:00 | 892.25 | 2025-05-05 14:15:00 | 911.75 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-04-24 13:30:00 | 891.90 | 2025-05-05 14:15:00 | 911.75 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-04-24 14:30:00 | 893.15 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-04-25 09:30:00 | 886.40 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2025-04-29 11:45:00 | 891.60 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-04-29 12:15:00 | 889.00 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-04-29 14:15:00 | 888.45 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-04-30 09:15:00 | 882.90 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2025-05-02 10:15:00 | 886.00 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-05-02 11:00:00 | 886.15 | 2025-05-06 09:15:00 | 925.55 | STOP_HIT | 1.00 | -4.45% |
| BUY | retest2 | 2025-07-29 14:30:00 | 1069.90 | 2025-07-31 09:15:00 | 1047.80 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-29 15:00:00 | 1074.00 | 2025-07-31 09:15:00 | 1047.80 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-07-30 12:15:00 | 1073.90 | 2025-07-31 09:15:00 | 1047.80 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-07-30 14:15:00 | 1069.60 | 2025-07-31 09:15:00 | 1047.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-08-01 09:15:00 | 1052.10 | 2025-08-01 09:15:00 | 1032.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-08-01 10:30:00 | 1057.10 | 2025-08-18 09:15:00 | 1162.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1057.90 | 2025-08-18 09:15:00 | 1163.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-04 10:45:00 | 1209.40 | 2026-02-12 11:15:00 | 1250.30 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2026-02-04 12:30:00 | 1209.70 | 2026-02-12 11:15:00 | 1250.30 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-02-06 09:15:00 | 1184.10 | 2026-02-12 11:15:00 | 1250.30 | STOP_HIT | 1.00 | -5.59% |
| SELL | retest2 | 2026-02-10 09:15:00 | 1205.80 | 2026-02-12 11:15:00 | 1250.30 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1234.00 | 2026-02-12 11:15:00 | 1250.30 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-12 10:00:00 | 1232.80 | 2026-02-12 11:15:00 | 1250.30 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1229.90 | 2026-02-24 13:15:00 | 1168.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 11:15:00 | 1229.20 | 2026-02-24 13:15:00 | 1167.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1229.90 | 2026-02-25 10:15:00 | 1208.10 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2026-02-16 11:15:00 | 1229.20 | 2026-02-25 10:15:00 | 1208.10 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2026-02-16 14:15:00 | 1214.70 | 2026-02-26 11:15:00 | 1234.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1212.60 | 2026-02-26 11:15:00 | 1234.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1211.20 | 2026-02-26 11:15:00 | 1234.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-02-25 15:15:00 | 1214.30 | 2026-02-26 11:15:00 | 1234.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1187.00 | 2026-03-02 09:15:00 | 1127.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1187.00 | 2026-03-09 09:15:00 | 1068.30 | TARGET_HIT | 0.50 | 10.00% |
