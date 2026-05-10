# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1655.00
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
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 10 |
| TARGET_HIT | 11 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 14
- **Target hits / Stop hits / Partials:** 11 / 15 / 10
- **Avg / median % per leg:** 3.60% / 5.00%
- **Sum % (uncompounded):** 129.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.96% | 14.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.96% | 14.8% |
| SELL (all) | 31 | 20 | 64.5% | 9 | 12 | 10 | 3.70% | 114.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 20 | 64.5% | 9 | 12 | 10 | 3.70% | 114.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 22 | 61.1% | 11 | 15 | 10 | 3.60% | 129.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1543.00 | 1577.25 | 1577.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1536.80 | 1572.93 | 1575.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1616.00 | 1572.87 | 1574.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1616.00 | 1572.87 | 1574.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1616.00 | 1572.87 | 1574.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 1573.90 | 1576.27 | 1576.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:30:00 | 1575.25 | 1576.23 | 1576.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:45:00 | 1575.00 | 1576.16 | 1576.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1564.55 | 1576.22 | 1576.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1560.30 | 1576.06 | 1576.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 1551.55 | 1575.82 | 1576.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 1546.30 | 1575.34 | 1576.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 1556.55 | 1571.96 | 1574.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1585.40 | 1566.49 | 1571.17 | SL hit (close>static) qty=1.00 sl=1584.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1585.40 | 1566.49 | 1571.17 | SL hit (close>static) qty=1.00 sl=1584.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1585.40 | 1566.49 | 1571.17 | SL hit (close>static) qty=1.00 sl=1584.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1641.90 | 1569.69 | 1572.28 | SL hit (close>static) qty=1.00 sl=1639.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1641.90 | 1569.69 | 1572.28 | SL hit (close>static) qty=1.00 sl=1639.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1641.90 | 1569.69 | 1572.28 | SL hit (close>static) qty=1.00 sl=1639.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1641.90 | 1569.69 | 1572.28 | SL hit (close>static) qty=1.00 sl=1639.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 1558.20 | 1571.88 | 1573.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1577.65 | 1571.52 | 1573.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:45:00 | 1575.60 | 1571.52 | 1573.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1571.50 | 1571.52 | 1573.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1562.20 | 1571.50 | 1573.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 1564.80 | 1571.39 | 1572.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:45:00 | 1566.60 | 1571.32 | 1572.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 1565.90 | 1569.89 | 1572.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1565.90 | 1569.85 | 1572.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1555.50 | 1569.85 | 1572.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 1555.55 | 1569.40 | 1571.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1555.70 | 1568.48 | 1571.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 1555.00 | 1568.26 | 1571.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1484.09 | 1557.68 | 1565.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1486.56 | 1557.68 | 1565.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1488.27 | 1557.68 | 1565.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1487.61 | 1557.68 | 1565.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1480.29 | 1544.98 | 1557.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.72 | 1544.98 | 1557.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.77 | 1544.98 | 1557.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.91 | 1544.98 | 1557.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.25 | 1544.98 | 1557.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1402.38 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1405.98 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1408.32 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1409.94 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1409.31 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1399.95 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1399.99 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1400.13 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-17 14:15:00 | 1399.50 | 1501.72 | 1530.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1491.20 | 1464.06 | 1500.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1491.20 | 1464.06 | 1500.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1494.20 | 1464.36 | 1499.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1488.60 | 1464.36 | 1499.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 13:15:00 | 1507.30 | 1465.15 | 1500.03 | SL hit (close>static) qty=1.00 sl=1503.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1486.60 | 1466.02 | 1500.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 1487.50 | 1466.20 | 1500.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:45:00 | 1490.00 | 1466.89 | 1499.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1499.70 | 1467.97 | 1498.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 1499.70 | 1467.97 | 1498.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1507.40 | 1468.36 | 1499.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-05 15:15:00 | 1507.40 | 1468.36 | 1499.02 | SL hit (close>static) qty=1.00 sl=1503.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 15:15:00 | 1507.40 | 1468.36 | 1499.02 | SL hit (close>static) qty=1.00 sl=1503.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 15:15:00 | 1507.40 | 1468.36 | 1499.02 | SL hit (close>static) qty=1.00 sl=1503.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 1510.40 | 1468.36 | 1499.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1498.30 | 1468.66 | 1499.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 1492.00 | 1468.87 | 1498.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 1417.40 | 1467.53 | 1495.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 1425.90 | 1422.22 | 1456.32 | SL hit (close>ema200) qty=0.50 sl=1422.22 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1568.40 | 1399.42 | 1398.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 1606.00 | 1431.71 | 1415.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-24 09:45:00 | 1408.75 | 2025-06-24 11:15:00 | 1391.35 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-24 10:15:00 | 1409.40 | 2025-06-24 11:15:00 | 1391.35 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-04 10:45:00 | 1414.55 | 2025-07-24 13:15:00 | 1553.09 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2025-07-04 14:00:00 | 1411.90 | 2025-07-25 09:15:00 | 1556.01 | TARGET_HIT | 1.00 | 10.21% |
| BUY | retest2 | 2025-10-13 14:15:00 | 1608.70 | 2025-10-15 09:15:00 | 1565.50 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-11-04 11:00:00 | 1573.90 | 2025-11-13 12:15:00 | 1585.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-04 12:30:00 | 1575.25 | 2025-11-13 12:15:00 | 1585.40 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-04 14:45:00 | 1575.00 | 2025-11-13 12:15:00 | 1585.40 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1564.55 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2025-11-06 10:30:00 | 1551.55 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2025-11-06 15:00:00 | 1546.30 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2025-11-10 12:30:00 | 1556.55 | 2025-11-19 09:15:00 | 1641.90 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2025-11-21 09:30:00 | 1558.20 | 2025-12-03 11:15:00 | 1484.09 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-03 11:15:00 | 1486.56 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-03 11:15:00 | 1488.27 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-03 11:15:00 | 1487.61 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-08 15:15:00 | 1480.29 | PARTIAL | 0.50 | 5.47% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-08 15:15:00 | 1477.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-08 15:15:00 | 1477.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-08 15:15:00 | 1477.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-08 15:15:00 | 1477.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:30:00 | 1558.20 | 2025-12-17 14:15:00 | 1402.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-17 14:15:00 | 1405.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-17 14:15:00 | 1408.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-17 14:15:00 | 1409.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-17 14:15:00 | 1409.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-17 14:15:00 | 1399.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-17 14:15:00 | 1399.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-17 14:15:00 | 1400.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-17 14:15:00 | 1399.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1488.60 | 2026-01-01 13:15:00 | 1507.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1486.60 | 2026-01-05 15:15:00 | 1507.40 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-02 09:45:00 | 1487.50 | 2026-01-05 15:15:00 | 1507.40 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-05 10:45:00 | 1490.00 | 2026-01-05 15:15:00 | 1507.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-06 10:30:00 | 1492.00 | 2026-01-09 09:15:00 | 1417.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:30:00 | 1492.00 | 2026-02-01 11:15:00 | 1425.90 | STOP_HIT | 0.50 | 4.43% |
