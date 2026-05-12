# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-11 15:15:00 (3605 bars)
- **Last close:** 1928.00
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
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 33
- **Target hits / Stop hits / Partials:** 1 / 33 / 2
- **Avg / median % per leg:** -2.12% / -2.06%
- **Sum % (uncompounded):** -76.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 1 | 8.3% | 1 | 11 | 0 | -1.81% | -21.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 1 | 8.3% | 1 | 11 | 0 | -1.81% | -21.7% |
| SELL (all) | 24 | 2 | 8.3% | 0 | 22 | 2 | -2.28% | -54.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 2 | 8.3% | 0 | 22 | 2 | -2.28% | -54.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 3 | 8.3% | 1 | 33 | 2 | -2.12% | -76.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 12:15:00 | 1318.00 | 1065.74 | 1064.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1321.70 | 1091.14 | 1078.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1310.00 | 1310.04 | 1238.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:00:00 | 1310.00 | 1310.04 | 1238.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1334.60 | 1389.49 | 1340.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:45:00 | 1338.00 | 1389.49 | 1340.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1337.00 | 1388.97 | 1340.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 1310.80 | 1388.97 | 1340.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1300.50 | 1387.29 | 1340.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 1298.90 | 1387.29 | 1340.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 1215.60 | 1308.86 | 1309.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1204.70 | 1284.45 | 1295.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 1243.00 | 1242.64 | 1266.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 1256.30 | 1242.64 | 1266.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1257.60 | 1243.43 | 1265.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 1259.30 | 1243.43 | 1265.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1259.40 | 1244.09 | 1265.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 1252.20 | 1244.28 | 1265.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 1252.30 | 1244.46 | 1265.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:00:00 | 1251.00 | 1244.57 | 1264.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1248.60 | 1244.44 | 1263.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1256.00 | 1244.96 | 1263.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 1262.80 | 1244.96 | 1263.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1272.70 | 1245.36 | 1263.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1272.70 | 1245.36 | 1263.17 | SL hit (close>static) qty=1.00 sl=1270.40 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 1392.00 | 1266.07 | 1266.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 1401.70 | 1267.42 | 1266.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1339.00 | 1341.13 | 1313.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:45:00 | 1337.80 | 1341.13 | 1313.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1321.60 | 1344.88 | 1318.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 1319.60 | 1344.88 | 1318.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1328.90 | 1344.73 | 1318.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 12:15:00 | 1331.20 | 1344.73 | 1318.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-15 13:15:00 | 1464.32 | 1380.09 | 1350.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1318.60 | 1432.18 | 1432.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1306.00 | 1413.82 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1397.40 | 1385.43 | 1405.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1397.40 | 1385.43 | 1405.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1402.00 | 1385.59 | 1405.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 1389.20 | 1386.23 | 1405.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1365.00 | 1387.05 | 1404.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1387.80 | 1386.83 | 1404.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 1382.50 | 1386.49 | 1403.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1392.00 | 1386.67 | 1403.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 1398.40 | 1386.67 | 1403.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1391.60 | 1386.76 | 1403.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 1454.30 | 1387.95 | 1403.31 | SL hit (close>static) qty=1.00 sl=1408.10 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1505.60 | 1416.39 | 1416.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 1510.10 | 1417.33 | 1416.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-16 11:15:00 | 1252.20 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-09-16 14:15:00 | 1252.30 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-17 10:00:00 | 1251.00 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-09-19 10:15:00 | 1248.60 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1246.00 | 2025-09-29 09:15:00 | 1183.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1252.20 | 2025-09-29 09:15:00 | 1189.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1246.00 | 2025-09-30 10:15:00 | 1256.60 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1252.20 | 2025-09-30 10:15:00 | 1256.60 | STOP_HIT | 0.50 | -0.35% |
| SELL | retest2 | 2025-09-30 11:30:00 | 1255.00 | 2025-10-01 14:15:00 | 1268.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1254.90 | 2025-10-06 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1242.90 | 2025-10-10 09:15:00 | 1268.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-10-03 13:30:00 | 1251.80 | 2025-10-10 09:15:00 | 1268.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-08 15:00:00 | 1255.00 | 2025-10-10 12:15:00 | 1283.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-09 15:00:00 | 1255.00 | 2025-10-10 12:15:00 | 1283.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-14 10:30:00 | 1250.70 | 2025-10-16 14:15:00 | 1278.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-15 09:30:00 | 1253.60 | 2025-10-16 14:15:00 | 1278.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-15 13:15:00 | 1254.40 | 2025-10-16 14:15:00 | 1278.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-10-20 10:00:00 | 1254.30 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.51% |
| SELL | retest2 | 2025-10-23 12:15:00 | 1256.30 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.34% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1252.60 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.65% |
| BUY | retest2 | 2025-11-24 12:15:00 | 1331.20 | 2025-12-15 13:15:00 | 1464.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 11:00:00 | 1334.00 | 2026-01-27 13:15:00 | 1306.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-27 14:45:00 | 1331.60 | 2026-01-29 13:15:00 | 1315.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-29 10:45:00 | 1330.00 | 2026-01-29 13:15:00 | 1315.70 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-11 10:30:00 | 1445.30 | 2026-02-13 09:15:00 | 1401.20 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-02-12 10:00:00 | 1442.20 | 2026-02-13 09:15:00 | 1401.20 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-02-18 12:30:00 | 1442.80 | 2026-02-19 15:15:00 | 1408.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-19 10:45:00 | 1443.70 | 2026-02-19 15:15:00 | 1408.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-02-20 09:45:00 | 1435.50 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2026-02-20 14:00:00 | 1444.00 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2026-03-11 10:00:00 | 1431.90 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2026-03-11 11:15:00 | 1445.10 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-04-09 10:30:00 | 1389.20 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1365.00 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -6.54% |
| SELL | retest2 | 2026-04-13 12:30:00 | 1387.80 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2026-04-15 11:00:00 | 1382.50 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -5.19% |
