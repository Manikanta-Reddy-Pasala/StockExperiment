# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3122 bars)
- **Last close:** 1493.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 21
- **Target hits / Stop hits / Partials:** 1 / 21 / 3
- **Avg / median % per leg:** -1.43% / -2.25%
- **Sum % (uncompounded):** -35.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.53% | -7.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.53% | -7.7% |
| SELL (all) | 20 | 4 | 20.0% | 1 | 16 | 3 | -1.41% | -28.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 4 | 20.0% | 1 | 16 | 3 | -1.41% | -28.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 4 | 16.0% | 1 | 21 | 3 | -1.43% | -35.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 1279.60 | 1333.17 | 1333.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 10:15:00 | 1257.00 | 1328.40 | 1330.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 1313.10 | 1310.69 | 1320.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 1313.10 | 1310.69 | 1320.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1308.00 | 1297.38 | 1310.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 1289.10 | 1305.64 | 1312.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1283.40 | 1304.97 | 1312.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 1289.90 | 1304.62 | 1311.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 1293.00 | 1304.35 | 1311.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1305.70 | 1303.75 | 1311.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 1312.30 | 1303.91 | 1311.04 | SL hit (close>static) qty=1.00 sl=1311.90 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 1409.90 | 1317.38 | 1317.00 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 15:15:00 | 1270.00 | 1316.74 | 1316.92 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 1331.40 | 1317.13 | 1317.11 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1293.00 | 1316.99 | 1317.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1282.10 | 1316.64 | 1316.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 1326.80 | 1310.94 | 1313.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 11:15:00 | 1326.80 | 1310.94 | 1313.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 1326.80 | 1310.94 | 1313.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:30:00 | 1323.80 | 1310.94 | 1313.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 1318.00 | 1311.01 | 1313.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:00:00 | 1314.50 | 1313.43 | 1314.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 1248.77 | 1304.66 | 1310.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1315.00 | 1292.29 | 1301.80 | SL hit (close>ema200) qty=0.50 sl=1292.29 alert=retest2 |

### Cycle 6 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1415.80 | 1247.29 | 1246.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1446.00 | 1249.26 | 1247.72 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-18 13:30:00 | 1330.00 | 2025-11-19 10:15:00 | 1317.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-20 13:30:00 | 1327.00 | 2025-12-03 09:15:00 | 1313.30 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-11-20 14:15:00 | 1328.70 | 2025-12-03 09:15:00 | 1313.30 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-28 13:00:00 | 1329.60 | 2025-12-03 09:15:00 | 1313.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-04 15:00:00 | 1351.00 | 2025-12-05 09:15:00 | 1306.60 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-01-12 15:00:00 | 1289.10 | 2026-01-19 12:15:00 | 1312.30 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1283.40 | 2026-01-19 12:15:00 | 1312.30 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-01-14 12:15:00 | 1289.90 | 2026-01-19 12:15:00 | 1312.30 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-14 13:45:00 | 1293.00 | 2026-01-19 12:15:00 | 1312.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-19 15:00:00 | 1283.20 | 2026-01-20 12:15:00 | 1219.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:00:00 | 1283.20 | 2026-01-21 09:15:00 | 1344.80 | STOP_HIT | 0.50 | -4.80% |
| SELL | retest2 | 2026-02-05 12:00:00 | 1314.50 | 2026-02-12 09:15:00 | 1248.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 12:00:00 | 1314.50 | 2026-02-23 09:15:00 | 1315.00 | STOP_HIT | 0.50 | -0.04% |
| SELL | retest2 | 2026-02-23 10:45:00 | 1311.00 | 2026-02-26 12:15:00 | 1344.50 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-02-26 09:30:00 | 1313.60 | 2026-02-26 12:15:00 | 1344.50 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-02-27 09:15:00 | 1302.00 | 2026-03-02 09:15:00 | 1236.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 1302.00 | 2026-03-09 09:15:00 | 1171.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-07 11:00:00 | 1214.10 | 2026-04-16 14:15:00 | 1241.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1208.00 | 2026-04-16 15:15:00 | 1244.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1217.30 | 2026-04-27 09:15:00 | 1249.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-04-13 15:15:00 | 1220.00 | 2026-04-27 09:15:00 | 1249.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-04-16 13:45:00 | 1223.80 | 2026-04-30 09:15:00 | 1303.20 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2026-04-16 14:30:00 | 1224.70 | 2026-04-30 09:15:00 | 1303.20 | STOP_HIT | 1.00 | -6.41% |
| SELL | retest2 | 2026-04-24 09:15:00 | 1224.00 | 2026-04-30 09:15:00 | 1303.20 | STOP_HIT | 1.00 | -6.47% |
| SELL | retest2 | 2026-04-24 10:30:00 | 1224.00 | 2026-04-30 09:15:00 | 1303.20 | STOP_HIT | 1.00 | -6.47% |
