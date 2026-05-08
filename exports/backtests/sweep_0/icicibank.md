# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1267.80
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
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 5 |
| PENDING | 11 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 0 / 7 / 1
- **Avg / median % per leg:** -0.67% / -0.80%
- **Sum % (uncompounded):** -5.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.90% | -1.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.90% | -1.8% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.60% | -3.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.60% | -3.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 2 | 25.0% | 0 | 7 | 1 | -0.67% | -5.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 1402.00 | 1438.59 | 1438.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 1399.30 | 1438.20 | 1438.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1424.40 | 1419.56 | 1427.19 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-15 15:15:00 | 1418.00 | 1419.70 | 1427.04 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-16 09:15:00 | 1421.20 | 1419.71 | 1427.01 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-09-17 15:15:00 | 1417.20 | 1419.83 | 1426.61 | ENTRY1 cross detected — sustain check pending (15m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1424.40 | 1419.87 | 1426.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1424.40 | 1419.87 | 1426.60 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-18 13:15:00 | 1420.30 | 1419.97 | 1426.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-18 14:15:00 | 1422.20 | 1420.00 | 1426.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-19 09:15:00 | 1405.10 | 1419.88 | 1426.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1403.80 | 1419.72 | 1426.26 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 1436.40 | 1392.90 | 1404.36 | SL hit (close>static) qty=1.00 sl=1432.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-20 09:15:00 | 1407.20 | 1393.48 | 1404.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 1401.80 | 1393.56 | 1404.52 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 1331.71 | 1377.11 | 1392.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 1382.00 | 1368.34 | 1384.78 | SL hit (close>ema200) qty=0.50 sl=1368.34 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 1412.90 | 1373.87 | 1375.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1403.00 | 1374.16 | 1375.93 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 1410.60 | 1377.69 | 1377.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1410.60 | 1377.69 | 1377.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 1420.00 | 1378.11 | 1377.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1382.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1382.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1382.16 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1354.50 | 1378.88 | 1378.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.30 | 1378.23 | 1378.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1377.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1377.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1377.04 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-30 10:15:00 | 1364.90 | 1375.48 | 1377.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1360.40 | 1375.33 | 1376.96 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1390.10 | 1371.24 | 1374.67 | SL hit (close>static) qty=1.00 sl=1378.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1403.70 | 1377.86 | 1377.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 15:15:00 | 1408.00 | 1378.43 | 1378.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1386.80 | 1391.26 | 1385.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 1388.70 | 1391.24 | 1385.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1388.70 | 1391.24 | 1385.62 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 1398.10 | 1391.31 | 1385.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1398.70 | 1391.38 | 1385.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1384.70 | 1392.13 | 1386.63 | SL hit (close<static) qty=1.00 sl=1384.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-25 09:15:00 | 1399.30 | 1392.14 | 1386.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 10:15:00 | 1394.90 | 1392.16 | 1386.73 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 1383.70 | 1393.09 | 1387.64 | SL hit (close<static) qty=1.00 sl=1384.80 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.25 | 1382.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.79 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 1290.30 | 1283.30 | 1317.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 1286.60 | 1283.33 | 1317.01 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 1350.30 | 1287.89 | 1316.92 | SL hit (close>static) qty=1.00 sl=1333.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-28 14:15:00 | 1290.00 | 1316.06 | 1325.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 1289.10 | 1315.79 | 1325.03 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-19 10:15:00 | 1403.80 | 2025-10-17 14:15:00 | 1436.40 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-20 10:15:00 | 1401.80 | 2025-11-06 09:15:00 | 1331.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-20 10:15:00 | 1401.80 | 2025-11-13 09:15:00 | 1382.00 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2026-01-09 10:15:00 | 1403.00 | 2026-01-12 14:15:00 | 1410.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-30 11:15:00 | 1360.40 | 2026-02-03 09:15:00 | 1390.10 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1398.70 | 2026-02-24 14:15:00 | 1384.70 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-25 10:15:00 | 1394.90 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-04-09 12:15:00 | 1286.60 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -4.95% |
