# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1734.80
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 17 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / Stop hits / Partials:** 0 / 14 / 0
- **Avg / median % per leg:** -1.00% / -2.01%
- **Sum % (uncompounded):** -14.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 0 | 14 | 0 | -1.00% | -14.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 0 | 14 | 0 | -1.00% | -14.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 3 | 21.4% | 0 | 14 | 0 | -1.00% | -14.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 1201.30 | 1155.00 | 1154.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1208.40 | 1150.06 | 1152.10 | Break + close above crossover candle high |

### Cycle 2 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 1214.90 | 1154.39 | 1154.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 1219.80 | 1156.17 | 1155.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1396.70 | 1397.66 | 1333.10 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 1336.70 | 1394.02 | 1338.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1336.70 | 1394.02 | 1338.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 1348.30 | 1391.45 | 1338.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-20 11:15:00 | 1344.00 | 1390.97 | 1338.64 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-20 14:15:00 | 1349.00 | 1389.55 | 1338.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 1353.90 | 1389.20 | 1338.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 1350.40 | 1388.40 | 1338.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 1358.90 | 1388.11 | 1338.98 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-01 15:15:00 | 1349.80 | 1411.31 | 1390.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1362.40 | 1410.83 | 1390.67 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1331.60 | 1402.35 | 1388.33 | SL hit (close<static) qty=1.00 sl=1332.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1331.60 | 1402.35 | 1388.33 | SL hit (close<static) qty=1.00 sl=1332.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1331.60 | 1402.35 | 1388.33 | SL hit (close<static) qty=1.00 sl=1332.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-19 10:15:00 | 1350.30 | 1372.29 | 1374.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:15:00 | 1357.50 | 1372.14 | 1374.51 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1356.10 | 1371.98 | 1374.41 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-19 14:15:00 | 1369.30 | 1371.78 | 1374.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 1370.00 | 1371.77 | 1374.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1342.80 | 1370.67 | 1373.52 | SL hit (close<static) qty=1.00 sl=1355.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1331.50 | 1366.94 | 1371.41 | SL hit (close<static) qty=1.00 sl=1332.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-09 09:15:00 | 1377.20 | 1350.82 | 1360.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1367.00 | 1350.99 | 1360.73 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 1407.20 | 1368.43 | 1368.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1407.20 | 1368.43 | 1368.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1409.40 | 1369.21 | 1368.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 10:15:00 | 1388.00 | 1391.40 | 1381.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 11:15:00 | 1380.00 | 1391.29 | 1381.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1380.00 | 1391.29 | 1381.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-29 13:15:00 | 1390.20 | 1391.24 | 1381.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 14:15:00 | 1382.40 | 1391.15 | 1381.54 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-29 15:15:00 | 1392.00 | 1391.16 | 1381.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1398.80 | 1391.24 | 1381.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-10-08 11:15:00 | 1391.00 | 1395.68 | 1385.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 1399.50 | 1395.71 | 1385.80 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 1367.60 | 1460.68 | 1463.02 | SL hit (close<static) qty=1.00 sl=1377.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 1367.60 | 1460.68 | 1463.02 | SL hit (close<static) qty=1.00 sl=1377.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-22 09:15:00 | 1417.30 | 1453.66 | 1459.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 1412.00 | 1453.24 | 1459.08 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-23 12:15:00 | 1355.20 | 1448.53 | 1456.42 | SL hit (close<static) qty=1.00 sl=1377.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-29 12:15:00 | 1390.50 | 1432.29 | 1447.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:15:00 | 1399.90 | 1431.97 | 1446.79 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1417.30 | 1431.82 | 1446.64 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-30 09:15:00 | 1422.40 | 1431.58 | 1446.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 1423.80 | 1431.51 | 1446.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-30 13:15:00 | 1422.50 | 1431.16 | 1445.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:15:00 | 1419.40 | 1431.04 | 1445.73 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-01 10:15:00 | 1425.30 | 1430.69 | 1445.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-01 11:15:00 | 1407.50 | 1430.46 | 1445.15 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 1385.20 | 1430.01 | 1444.85 | SL hit (close<static) qty=1.00 sl=1397.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 1385.20 | 1430.01 | 1444.85 | SL hit (close<static) qty=1.00 sl=1397.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 1372.50 | 1429.44 | 1444.49 | SL hit (close<static) qty=1.00 sl=1377.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1506.10 | 1426.79 | 1442.29 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 1562.90 | 1456.70 | 1456.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1562.90 | 1456.70 | 1456.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1568.50 | 1457.81 | 1456.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 1502.90 | 1417.20 | 1434.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 1502.00 | 1418.05 | 1434.55 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 1572.10 | 1449.50 | 1449.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 1572.10 | 1449.50 | 1449.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 1598.10 | 1450.98 | 1449.90 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 15:15:00 | 1353.90 | 2025-08-07 10:15:00 | 1331.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-06-23 11:15:00 | 1358.90 | 2025-08-07 10:15:00 | 1331.60 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1362.40 | 2025-08-07 10:15:00 | 1331.60 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-08-19 11:15:00 | 1357.50 | 2025-08-22 10:15:00 | 1342.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-08-19 15:15:00 | 1370.00 | 2025-08-26 09:15:00 | 1331.50 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1367.00 | 2025-09-17 14:15:00 | 1407.20 | STOP_HIT | 1.00 | 2.94% |
| BUY | retest2 | 2025-09-30 09:15:00 | 1398.80 | 2026-01-20 14:15:00 | 1367.60 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-10-08 12:15:00 | 1399.50 | 2026-01-20 14:15:00 | 1367.60 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-01-22 10:15:00 | 1412.00 | 2026-01-23 12:15:00 | 1355.20 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2026-01-29 13:15:00 | 1399.90 | 2026-02-01 12:15:00 | 1385.20 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-30 10:15:00 | 1423.80 | 2026-02-01 12:15:00 | 1385.20 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-01-30 14:15:00 | 1419.40 | 2026-02-01 13:15:00 | 1372.50 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2026-02-03 10:15:00 | 1506.10 | 2026-02-09 10:15:00 | 1562.90 | STOP_HIT | 1.00 | 3.77% |
| BUY | retest2 | 2026-04-15 10:15:00 | 1502.00 | 2026-04-20 15:15:00 | 1572.10 | STOP_HIT | 1.00 | 4.67% |
