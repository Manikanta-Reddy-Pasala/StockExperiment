# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1475.30
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 18
- **Target hits / Stop hits / Partials:** 2 / 18 / 0
- **Avg / median % per leg:** -0.27% / -1.07%
- **Sum % (uncompounded):** -5.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 2 | 7 | 0 | 1.59% | 14.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 2 | 7 | 0 | 1.59% | 14.3% |
| SELL (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.78% | -19.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.78% | -19.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 2 | 10.0% | 2 | 18 | 0 | -0.27% | -5.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 1115.85 | 1187.85 | 1187.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1111.90 | 1169.11 | 1177.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1161.20 | 1146.25 | 1163.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1166.00 | 1146.45 | 1163.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 1166.00 | 1146.45 | 1163.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1160.90 | 1146.59 | 1163.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 1158.20 | 1146.87 | 1163.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1157.60 | 1146.97 | 1162.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1171.60 | 1147.22 | 1162.88 | SL hit (close>static) qty=1.00 sl=1167.80 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.20 | 1171.86 | 1171.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1186.50 | 1187.13 | 1180.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 1186.50 | 1187.13 | 1180.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1171.90 | 1187.00 | 1180.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 1171.90 | 1187.00 | 1180.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1168.30 | 1186.81 | 1180.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1168.30 | 1186.81 | 1180.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1182.60 | 1185.53 | 1179.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1178.00 | 1185.53 | 1179.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1177.50 | 1185.42 | 1179.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 1177.50 | 1185.42 | 1179.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1180.00 | 1185.36 | 1179.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1183.50 | 1185.36 | 1179.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 11:15:00 | 1173.30 | 1185.13 | 1179.90 | SL hit (close<static) qty=1.00 sl=1176.30 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1234.50 | 1278.90 | 1279.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.30 | 1277.41 | 1278.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:30:00 | 1218.90 | 1222.43 | 1242.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 1215.40 | 1222.04 | 1242.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:00:00 | 1220.10 | 1221.97 | 1241.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:45:00 | 1220.00 | 1221.97 | 1241.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1240.40 | 1222.74 | 1241.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 1240.40 | 1222.74 | 1241.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1248.60 | 1223.00 | 1241.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:00:00 | 1248.60 | 1223.00 | 1241.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 1250.00 | 1223.26 | 1241.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1237.70 | 1223.26 | 1241.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 13:15:00 | 1253.60 | 1224.69 | 1241.22 | SL hit (close>static) qty=1.00 sl=1250.60 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1416.30 | 1264.47 | 1259.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-18 13:45:00 | 1158.20 | 2025-08-20 09:15:00 | 1171.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1157.60 | 2025-08-20 09:15:00 | 1171.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-22 15:00:00 | 1160.40 | 2025-08-26 09:15:00 | 1169.90 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-26 15:15:00 | 1159.20 | 2025-09-01 11:15:00 | 1168.80 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-08-29 10:00:00 | 1150.50 | 2025-09-01 11:15:00 | 1168.80 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-08-29 13:45:00 | 1154.70 | 2025-09-01 11:15:00 | 1168.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-25 09:15:00 | 1183.50 | 2025-09-25 11:15:00 | 1173.30 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-06 13:00:00 | 1180.60 | 2025-10-07 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-10-06 14:30:00 | 1180.20 | 2025-10-07 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-07 09:30:00 | 1181.00 | 2025-10-07 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-07 12:15:00 | 1185.00 | 2025-10-08 09:15:00 | 1166.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-09 14:00:00 | 1185.00 | 2025-10-14 12:15:00 | 1172.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-13 12:30:00 | 1185.20 | 2025-10-14 12:15:00 | 1172.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1188.70 | 2025-10-17 09:15:00 | 1307.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1201.30 | 2026-01-06 09:15:00 | 1321.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-08 11:30:00 | 1218.90 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2026-04-09 09:45:00 | 1215.40 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-04-09 12:00:00 | 1220.10 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2026-04-09 12:45:00 | 1220.00 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1237.70 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -1.28% |
