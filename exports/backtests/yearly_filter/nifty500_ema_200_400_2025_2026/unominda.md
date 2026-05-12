# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1179.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 17
- **Target hits / Stop hits / Partials:** 3 / 20 / 3
- **Avg / median % per leg:** 0.45% / -1.42%
- **Sum % (uncompounded):** 11.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 2 | 5 | 0 | 1.31% | 9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 2 | 5 | 0 | 1.31% | 9.2% |
| SELL (all) | 19 | 7 | 36.8% | 1 | 15 | 3 | 0.14% | 2.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 7 | 36.8% | 1 | 15 | 3 | 0.14% | 2.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 9 | 34.6% | 3 | 20 | 3 | 0.45% | 11.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 992.30 | 917.77 | 917.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 1005.00 | 921.38 | 919.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1068.80 | 1069.13 | 1031.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 1068.80 | 1069.13 | 1031.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1049.20 | 1076.70 | 1049.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 1046.90 | 1076.70 | 1049.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1048.80 | 1076.43 | 1049.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 1048.00 | 1076.43 | 1049.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1061.90 | 1076.28 | 1049.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:30:00 | 1069.90 | 1076.17 | 1049.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 1074.00 | 1076.17 | 1049.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 12:15:00 | 1073.90 | 1075.85 | 1050.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 1069.60 | 1075.74 | 1050.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1047.80 | 1075.32 | 1050.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1047.80 | 1075.32 | 1050.69 | SL hit (close<static) qty=1.00 sl=1047.90 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-16 15:15:00)

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
| SELL | retest2 | 2025-05-14 10:30:00 | 995.60 | 2025-05-16 12:15:00 | 992.30 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-05-14 11:45:00 | 991.85 | 2025-05-16 12:15:00 | 992.30 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-05-15 15:15:00 | 987.00 | 2025-05-16 12:15:00 | 992.30 | STOP_HIT | 1.00 | -0.54% |
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
