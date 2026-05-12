# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
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
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 14 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 12 / 9
- **Target hits / Stop hits / Partials:** 3 / 13 / 5
- **Avg / median % per leg:** 2.02% / 0.16%
- **Sum % (uncompounded):** 42.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.06% | -9.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.06% | -9.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 12 | 66.7% | 3 | 10 | 5 | 2.87% | 51.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 12 | 66.7% | 3 | 10 | 5 | 2.87% | 51.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.06% | -9.2% |
| retest2 (combined) | 18 | 12 | 66.7% | 3 | 10 | 5 | 2.87% | 51.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-16 12:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 12:00:00 | 1482.80 | 1481.30 | 1439.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 14:00:00 | 1485.00 | 1481.33 | 1439.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:45:00 | 1481.90 | 1481.41 | 1440.80 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1446.90 | 1480.53 | 1442.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1443.10 | 1480.53 | 1442.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1445.50 | 1480.18 | 1442.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 1444.90 | 1480.18 | 1442.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1437.90 | 1479.76 | 1442.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1437.90 | 1479.76 | 1442.90 | SL hit (close<ema400) qty=1.00 sl=1442.90 alert=retest1 |

### Cycle 2 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.44 | 1423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1380.50 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 1415.70 | 1413.28 | 1417.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1419.50 | 1413.36 | 1417.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1419.50 | 1413.36 | 1417.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1421.00 | 1413.44 | 1417.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1412.50 | 1413.44 | 1417.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 1422.90 | 1413.60 | 1418.02 | SL hit (close>static) qty=1.00 sl=1421.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.01 | 1398.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1483.00 | 1401.36 | 1400.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.17 | 1511.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 1539.30 | 1541.17 | 1511.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.51 | 1520.74 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.20 | 1501.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1370.70 | 1457.67 | 1477.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1449.75 | 1471.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:00:00 | 1459.10 | 1449.75 | 1471.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 1462.00 | 1450.57 | 1468.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1462.60 | 1451.01 | 1468.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 1462.20 | 1452.02 | 1468.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1388.90 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.47 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.09 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1315.80 | 1427.64 | 1448.33 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-16 12:00:00 | 1482.80 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2025-07-16 14:00:00 | 1485.00 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest1 | 2025-07-17 10:45:00 | 1481.90 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1412.50 | 2025-08-20 10:15:00 | 1422.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-08-20 13:15:00 | 1419.10 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-20 14:15:00 | 1417.20 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1416.60 | 2025-09-01 09:15:00 | 1345.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 12:15:00 | 1412.60 | 2025-09-01 09:15:00 | 1341.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1416.60 | 2025-09-12 10:15:00 | 1394.00 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2025-08-22 12:15:00 | 1412.60 | 2025-09-12 10:15:00 | 1394.00 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-08-22 14:30:00 | 1410.50 | 2025-09-17 09:15:00 | 1410.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-08-25 09:45:00 | 1410.10 | 2025-10-17 09:15:00 | 1407.80 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-08-25 11:00:00 | 1411.10 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-16 13:45:00 | 1399.90 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-22 14:15:00 | 1394.60 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-02-27 09:15:00 | 1388.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-02-27 09:15:00 | 1389.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-02-27 09:15:00 | 1389.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-03-04 09:15:00 | 1315.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-03-04 09:15:00 | 1316.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-03-04 09:15:00 | 1315.98 | TARGET_HIT | 0.50 | 10.00% |
