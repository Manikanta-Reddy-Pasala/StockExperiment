# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2025-04-21 09:15:00 → 2026-05-08 15:15:00 (1822 bars)
- **Last close:** 1267.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 0 / 14 / 2
- **Avg / median % per leg:** -0.79% / -1.19%
- **Sum % (uncompounded):** -12.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.52% | -9.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.52% | -9.1% |
| SELL (all) | 10 | 4 | 40.0% | 0 | 8 | 2 | -0.36% | -3.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 0 | 8 | 2 | -0.36% | -3.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 4 | 25.0% | 0 | 14 | 2 | -0.79% | -12.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 1475.50 | 1432.35 | 1432.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 1479.90 | 1434.81 | 1433.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 1452.70 | 1455.02 | 1445.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:45:00 | 1450.90 | 1455.02 | 1445.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1448.20 | 1454.95 | 1445.60 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 1416.30 | 1440.37 | 1440.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 1415.00 | 1440.11 | 1440.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1424.40 | 1419.47 | 1427.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 1424.40 | 1419.47 | 1427.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1424.40 | 1419.80 | 1426.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:45:00 | 1420.80 | 1419.91 | 1426.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1412.40 | 1419.96 | 1426.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 1349.76 | 1406.29 | 1418.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1389.80 | 1388.79 | 1403.79 | SL hit (close>ema200) qty=0.50 sl=1388.79 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1410.60 | 1377.69 | 1377.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 1420.00 | 1378.11 | 1377.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1382.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1382.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1382.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1362.10 | 1386.14 | 1382.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1370.20 | 1385.98 | 1382.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 11:45:00 | 1372.50 | 1385.83 | 1382.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:45:00 | 1380.10 | 1385.62 | 1382.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1348.60 | 1384.81 | 1381.77 | SL hit (close<static) qty=1.00 sl=1360.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1354.50 | 1378.88 | 1378.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.30 | 1378.23 | 1378.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1377.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1377.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1377.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 1378.00 | 1375.40 | 1377.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1383.20 | 1375.48 | 1377.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 1384.10 | 1375.48 | 1377.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 1385.00 | 1375.58 | 1377.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1375.10 | 1375.58 | 1377.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1390.10 | 1371.24 | 1374.69 | SL hit (close>static) qty=1.00 sl=1386.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1403.70 | 1377.86 | 1377.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 15:15:00 | 1408.00 | 1378.43 | 1378.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1386.80 | 1391.26 | 1385.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 1386.80 | 1391.26 | 1385.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1388.70 | 1391.24 | 1385.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1389.30 | 1391.24 | 1385.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1398.10 | 1391.31 | 1385.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1400.80 | 1391.31 | 1385.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 1404.80 | 1391.58 | 1386.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 1400.40 | 1392.06 | 1386.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1400.00 | 1392.14 | 1386.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1389.40 | 1393.23 | 1387.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1386.40 | 1393.23 | 1387.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1383.70 | 1393.09 | 1387.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 1383.70 | 1393.09 | 1387.65 | SL hit (close<static) qty=1.00 sl=1384.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.25 | 1382.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 1291.50 | 1283.11 | 1317.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 1291.90 | 1283.23 | 1317.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:00:00 | 1290.30 | 1283.30 | 1317.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 1350.30 | 1287.89 | 1316.92 | SL hit (close>static) qty=1.00 sl=1333.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-18 13:45:00 | 1420.80 | 2025-09-29 12:15:00 | 1349.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 13:45:00 | 1420.80 | 2025-10-15 09:15:00 | 1389.80 | STOP_HIT | 0.50 | 2.18% |
| SELL | retest2 | 2025-09-19 09:15:00 | 1412.40 | 2025-10-17 14:15:00 | 1436.40 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-10-20 09:15:00 | 1408.00 | 2025-11-04 11:15:00 | 1337.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-20 09:15:00 | 1408.00 | 2025-11-13 09:15:00 | 1382.00 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2026-01-07 13:30:00 | 1419.00 | 2026-01-08 09:15:00 | 1434.40 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-19 11:45:00 | 1372.50 | 2026-01-21 09:15:00 | 1348.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-19 13:45:00 | 1380.10 | 2026-01-21 09:15:00 | 1348.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1375.10 | 2026-02-03 09:15:00 | 1390.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1400.80 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-02-23 09:15:00 | 1404.80 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-23 14:15:00 | 1400.40 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-25 09:30:00 | 1400.00 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-09 09:45:00 | 1291.50 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-04-09 10:30:00 | 1291.90 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2026-04-09 12:00:00 | 1290.30 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -4.65% |
