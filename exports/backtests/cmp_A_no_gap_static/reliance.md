# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1436.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 10 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 6 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 3.18% / 5.00%
- **Sum % (uncompounded):** 28.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.86% | -2.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.86% | -2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 2 | 3 | 3 | 3.94% | 31.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.27% | -1.3% |
| SELL @ 3rd Alert (retest2) | 7 | 6 | 85.7% | 2 | 2 | 3 | 4.68% | 32.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.06% | -4.1% |
| retest2 (combined) | 7 | 6 | 85.7% | 2 | 2 | 3 | 4.68% | 32.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 1396.20 | 1448.40 | 1448.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1389.70 | 1447.82 | 1448.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 1415.10 | 1413.89 | 1427.70 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-20 14:15:00 | 1413.00 | 1414.34 | 1427.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 15:15:00 | 1412.00 | 1414.31 | 1427.31 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1429.90 | 1414.47 | 1427.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 1429.90 | 1414.47 | 1427.32 | SL hit (close>ema400) qty=1.00 sl=1427.32 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-22 10:15:00 | 1414.40 | 1415.23 | 1427.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 1413.60 | 1415.22 | 1427.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 1342.92 | 1408.50 | 1421.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 1394.00 | 1392.37 | 1408.64 | SL hit (close>ema200) qty=0.50 sl=1392.37 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-17 12:15:00 | 1411.50 | 1384.22 | 1394.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:15:00 | 1414.00 | 1384.52 | 1394.38 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1464.80 | 1385.97 | 1394.96 | SL hit (close>static) qty=1.00 sl=1431.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1483.00 | 1403.00 | 1402.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 12:15:00 | 1486.50 | 1409.14 | 1406.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.17 | 1512.07 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-31 09:15:00 | 1548.00 | 1541.24 | 1512.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1555.00 | 1541.38 | 1512.60 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.51 | 1521.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 1510.50 | 1550.11 | 1521.05 | SL hit (close<ema400) qty=1.00 sl=1521.05 alert=retest1 |

### Cycle 3 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 1390.60 | 1502.38 | 1502.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 1384.00 | 1501.21 | 1501.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1449.75 | 1471.41 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 1437.50 | 1449.54 | 1469.85 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 10:15:00 | 1448.00 | 1449.53 | 1469.74 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.87 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-10 11:15:00 | 1459.80 | 1450.57 | 1468.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:15:00 | 1460.20 | 1450.67 | 1468.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-12 10:15:00 | 1461.60 | 1452.02 | 1468.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 1457.10 | 1452.07 | 1468.35 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1387.19 | 1432.96 | 1451.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1384.24 | 1432.96 | 1451.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1314.18 | 1427.64 | 1448.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1311.39 | 1427.64 | 1448.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-05 10:15:00 | 1452.90 | 1378.28 | 1392.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1460.10 | 1379.09 | 1392.70 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 13:15:00 | 1460.40 | 1380.76 | 1393.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-05 14:15:00 | 1464.40 | 1381.59 | 1393.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-05 15:15:00 | 1462.50 | 1382.40 | 1394.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1455.50 | 1383.13 | 1394.41 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-20 15:15:00 | 1412.00 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-22 11:15:00 | 1413.60 | 2025-09-01 09:15:00 | 1342.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 11:15:00 | 1413.60 | 2025-09-12 10:15:00 | 1394.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-10-17 13:15:00 | 1414.00 | 2025-10-20 09:15:00 | 1464.80 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest1 | 2025-12-31 10:15:00 | 1555.00 | 2026-01-06 10:15:00 | 1510.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2026-02-10 12:15:00 | 1460.20 | 2026-03-02 09:15:00 | 1387.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1457.10 | 2026-03-02 09:15:00 | 1384.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 12:15:00 | 1460.20 | 2026-03-04 09:15:00 | 1314.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1457.10 | 2026-03-04 09:15:00 | 1311.39 | TARGET_HIT | 0.50 | 10.00% |
