# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1493.20
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
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 9 |
| TARGET_HIT | 10 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 12
- **Target hits / Stop hits / Partials:** 10 / 12 / 9
- **Avg / median % per leg:** 3.82% / 5.00%
- **Sum % (uncompounded):** 118.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 5 | 1 | 0 | 7.96% | 47.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 5 | 83.3% | 5 | 1 | 0 | 7.96% | 47.8% |
| SELL (all) | 25 | 14 | 56.0% | 5 | 11 | 9 | 2.83% | 70.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 14 | 56.0% | 5 | 11 | 9 | 2.83% | 70.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 31 | 19 | 61.3% | 10 | 12 | 9 | 3.82% | 118.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1255.30 | 1350.52 | 1350.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1252.70 | 1349.55 | 1350.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 1313.10 | 1310.90 | 1326.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 1313.10 | 1310.90 | 1326.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1317.50 | 1297.83 | 1315.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 1324.00 | 1297.83 | 1315.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1305.10 | 1297.91 | 1315.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1303.50 | 1298.76 | 1315.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 1301.30 | 1298.79 | 1315.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1322.30 | 1299.19 | 1315.82 | SL hit (close>static) qty=1.00 sl=1318.20 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1415.80 | 1247.29 | 1247.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1446.00 | 1249.26 | 1248.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 09:15:00 | 1133.60 | 2025-06-20 12:15:00 | 1108.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-06-23 09:15:00 | 1135.60 | 2025-07-01 14:15:00 | 1249.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 10:00:00 | 1124.90 | 2025-07-01 14:15:00 | 1237.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 14:45:00 | 1125.70 | 2025-07-01 14:15:00 | 1238.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1277.70 | 2025-08-22 11:15:00 | 1405.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 10:30:00 | 1282.00 | 2025-10-24 10:15:00 | 1410.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 1303.50 | 2026-01-06 11:15:00 | 1322.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-01-06 10:00:00 | 1301.30 | 2026-01-06 11:15:00 | 1322.30 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-07 11:45:00 | 1304.10 | 2026-01-07 12:15:00 | 1322.20 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1302.00 | 2026-01-20 12:15:00 | 1236.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:45:00 | 1299.60 | 2026-01-20 12:15:00 | 1234.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1302.00 | 2026-01-21 09:15:00 | 1344.80 | STOP_HIT | 0.50 | -3.29% |
| SELL | retest2 | 2026-01-19 13:45:00 | 1299.60 | 2026-01-21 09:15:00 | 1344.80 | STOP_HIT | 0.50 | -3.48% |
| SELL | retest2 | 2026-01-28 09:30:00 | 1299.70 | 2026-01-30 09:15:00 | 1340.60 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-02-01 09:15:00 | 1306.20 | 2026-02-02 12:15:00 | 1240.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:45:00 | 1304.00 | 2026-02-02 12:15:00 | 1238.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 1306.20 | 2026-02-03 11:15:00 | 1326.80 | STOP_HIT | 0.50 | -1.58% |
| SELL | retest2 | 2026-02-01 09:45:00 | 1304.00 | 2026-02-03 11:15:00 | 1326.80 | STOP_HIT | 0.50 | -1.75% |
| SELL | retest2 | 2026-02-06 10:15:00 | 1284.90 | 2026-02-26 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-02-09 09:45:00 | 1280.90 | 2026-02-26 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-02-09 11:30:00 | 1284.10 | 2026-03-02 09:15:00 | 1228.83 | PARTIAL | 0.50 | 4.30% |
| SELL | retest2 | 2026-02-09 12:30:00 | 1278.50 | 2026-03-02 10:15:00 | 1220.65 | PARTIAL | 0.50 | 4.52% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1290.00 | 2026-03-02 10:15:00 | 1216.86 | PARTIAL | 0.50 | 5.67% |
| SELL | retest2 | 2026-02-23 15:15:00 | 1285.00 | 2026-03-02 10:15:00 | 1219.89 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-02-27 09:30:00 | 1293.50 | 2026-03-02 10:15:00 | 1214.58 | PARTIAL | 0.50 | 6.10% |
| SELL | retest2 | 2026-02-09 11:30:00 | 1284.10 | 2026-03-09 09:15:00 | 1156.41 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest2 | 2026-02-09 12:30:00 | 1278.50 | 2026-03-09 09:15:00 | 1152.81 | TARGET_HIT | 0.50 | 9.83% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1290.00 | 2026-03-09 09:15:00 | 1155.69 | TARGET_HIT | 0.50 | 10.41% |
| SELL | retest2 | 2026-02-23 15:15:00 | 1285.00 | 2026-03-09 09:15:00 | 1150.65 | TARGET_HIT | 0.50 | 10.46% |
| SELL | retest2 | 2026-02-27 09:30:00 | 1293.50 | 2026-03-09 09:15:00 | 1164.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-30 10:30:00 | 1293.70 | 2026-04-30 11:15:00 | 1315.00 | STOP_HIT | 1.00 | -1.65% |
