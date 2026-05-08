# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1760.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 5 |
| PENDING | 6 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.88% / 1.67%
- **Sum % (uncompounded):** 6.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.61% | 9.6% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.35% | 13.4% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.88% | -3.8% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.51% | -3.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.51% | -3.5% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.35% | 13.4% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.42% | -7.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 1373.80 | 1411.21 | 1411.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 1359.60 | 1408.69 | 1410.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 1353.50 | 1350.76 | 1370.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1377.20 | 1351.12 | 1369.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1377.20 | 1351.12 | 1369.89 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 1440.70 | 1381.97 | 1381.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 1452.00 | 1403.15 | 1395.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 1417.50 | 1422.19 | 1407.15 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-29 09:15:00 | 1435.70 | 1421.90 | 1408.01 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:15:00 | 1451.90 | 1422.19 | 1408.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 10:15:00 | 1450.80 | 1431.34 | 1415.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 11:15:00 | 1452.60 | 1431.55 | 1416.17 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:15:00 | 1524.50 | 1471.46 | 1446.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:15:00 | 1525.23 | 1471.46 | 1446.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 1476.90 | 1484.87 | 1459.39 | SL hit (close<ema200) qty=0.50 sl=1484.87 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 1476.90 | 1484.87 | 1459.39 | SL hit (close<ema200) qty=0.50 sl=1484.87 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1466.90 | 1493.53 | 1474.42 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-01 11:15:00 | 1488.70 | 1488.34 | 1473.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-01 12:15:00 | 1481.70 | 1488.28 | 1473.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-02 09:15:00 | 1494.50 | 1488.11 | 1473.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 1490.60 | 1488.13 | 1474.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-05 12:15:00 | 1492.50 | 1488.13 | 1474.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:15:00 | 1489.80 | 1488.15 | 1474.75 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 1462.20 | 1486.96 | 1474.93 | SL hit (close<static) qty=1.00 sl=1465.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 1462.20 | 1486.96 | 1474.93 | SL hit (close<static) qty=1.00 sl=1465.10 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 1402.10 | 1465.74 | 1465.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1396.30 | 1463.83 | 1465.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1442.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1442.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1442.25 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1562.90 | 1456.70 | 1456.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1568.50 | 1457.81 | 1457.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.48 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.01 | 1476.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.47 | 1461.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.30 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 14:15:00 | 1452.10 | 1405.56 | 1430.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:15:00 | 1452.00 | 1406.02 | 1431.01 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1502.90 | 1417.20 | 1434.26 | SL hit (close>static) qty=1.00 sl=1487.80 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 1572.10 | 1449.50 | 1449.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 1598.10 | 1450.98 | 1449.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-29 10:15:00 | 1451.90 | 2025-11-28 11:15:00 | 1524.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-11-07 11:15:00 | 1452.60 | 2025-11-28 11:15:00 | 1525.23 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-29 10:15:00 | 1451.90 | 2025-12-08 13:15:00 | 1476.90 | STOP_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2025-11-07 11:15:00 | 1452.60 | 2025-12-08 13:15:00 | 1476.90 | STOP_HIT | 0.50 | 1.67% |
| BUY | retest2 | 2026-01-02 10:15:00 | 1490.60 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-01-05 13:15:00 | 1489.80 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-08 15:15:00 | 1452.00 | 2026-04-15 09:15:00 | 1502.90 | STOP_HIT | 1.00 | -3.51% |
