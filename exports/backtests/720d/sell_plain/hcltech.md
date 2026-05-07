# HCLTECH (HCLTECH)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1182.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 13 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 2 |
| ENTRY2 | 6 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 5
- **Avg / median % per leg:** 9.67% / 14.68%
- **Sum % (uncompounded):** 77.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 7 | 87.5% | 0 | 3 | 5 | 9.67% | 77.4% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 5.63% | 16.9% |
| SELL @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 0 | 1 | 4 | 12.10% | 60.5% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 5.63% | 16.9% |
| retest2 (combined) | 5 | 5 | 100.0% | 0 | 1 | 4 | 12.10% | 60.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 14:15:00 | 1710.95 | 1860.66 | 1861.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 11:15:00 | 1709.25 | 1854.92 | 1858.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 09:15:00 | 1643.55 | 1614.37 | 1680.03 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-27 12:15:00 | 1619.95 | 1616.84 | 1675.97 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-27 13:15:00 | 1620.65 | 1616.88 | 1675.70 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-28 09:15:00 | 1611.00 | 1617.16 | 1674.96 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:15:00 | 1616.20 | 1617.14 | 1674.38 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-07 09:15:00 | 1373.77 | 1582.82 | 1647.53 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1569.00 | 1510.51 | 1587.28 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 1569.00 | 1510.51 | 1587.28 | SL hit (close>ema200) qty=0.50 sl=1510.51 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-18 09:15:00 | 1534.70 | 1658.13 | 1653.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 1542.30 | 1655.79 | 1652.52 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-07-21 09:15:00 | 1537.80 | 1650.41 | 1649.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Stop hit — per-position SL triggered | 2025-07-21 10:15:00 | 1534.40 | 1649.25 | 1649.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1534.40 | 1649.25 | 1649.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1529.50 | 1646.92 | 1648.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1483.10 | 1476.51 | 1514.59 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-09-19 12:15:00 | 1470.30 | 1478.04 | 1511.94 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 14:15:00 | 1465.90 | 1477.87 | 1511.52 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1467.20 | 1443.05 | 1478.64 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-09 10:15:00 | 1481.00 | 1443.42 | 1478.65 | SL hit (close>ema400) qty=1.00 sl=1478.65 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 1405.80 | 1633.78 | 1630.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-13 11:15:00 | 1463.00 | 1630.24 | 1628.33 | ENTRY2 sustain failed after 120m |

### Cycle 3 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1454.70 | 1625.22 | 1625.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1450.70 | 1587.14 | 1605.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.18 | 1457.06 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1437.90 | 1399.27 | 1454.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.27 | 1454.12 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1433.70 | 1406.19 | 1453.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 11:15:00 | 1428.30 | 1406.64 | 1453.74 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1427.70 | 1408.38 | 1453.46 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 1422.90 | 1408.67 | 1453.16 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-17 10:15:00 | 1432.90 | 1414.71 | 1452.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:15:00 | 1434.70 | 1415.09 | 1451.90 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-20 09:15:00 | 1431.30 | 1415.99 | 1451.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-20 11:15:00 | 1438.90 | 1416.38 | 1451.47 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-20 13:15:00 | 1431.70 | 1416.74 | 1451.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 1424.40 | 1416.94 | 1451.06 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1449.40 | 1417.71 | 1450.94 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-21 12:15:00 | 1436.60 | 1417.90 | 1450.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-21 13:15:00 | 1442.00 | 1418.14 | 1450.82 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-21 15:15:00 | 1439.00 | 1418.57 | 1450.72 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1305.20 | 1417.45 | 1449.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 10:15:00 | 1219.50 | 1397.45 | 1437.29 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 11:15:00 | 1214.05 | 1395.60 | 1436.17 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 11:15:00 | 1210.74 | 1395.60 | 1436.17 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 12:15:00 | 1209.47 | 1393.68 | 1435.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-03-28 11:15:00 | 1616.20 | 2025-04-07 09:15:00 | 1373.77 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2025-03-28 11:15:00 | 1616.20 | 2025-04-23 09:15:00 | 1569.00 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2025-07-18 11:15:00 | 1542.30 | 2025-07-21 10:15:00 | 1534.40 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest1 | 2025-09-19 14:15:00 | 1465.90 | 2025-10-09 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-04-10 11:15:00 | 1428.30 | 2026-04-24 10:15:00 | 1219.50 | PARTIAL | 0.50 | 14.62% |
| SELL | retest2 | 2026-04-13 11:15:00 | 1422.90 | 2026-04-24 11:15:00 | 1214.05 | PARTIAL | 0.50 | 14.68% |
| SELL | retest2 | 2026-04-17 12:15:00 | 1434.70 | 2026-04-24 11:15:00 | 1210.74 | PARTIAL | 0.50 | 15.61% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-24 12:15:00 | 1209.47 | PARTIAL | 0.50 | 15.09% |
