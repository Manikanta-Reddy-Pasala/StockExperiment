# TBO Tek Ltd. (TBOTEK)

## Backtest Summary

- **Window:** 2024-05-15 09:15:00 → 2026-05-08 15:15:00 (3430 bars)
- **Last close:** 1227.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 0
- **Avg / median % per leg:** -2.15% / -2.88%
- **Sum % (uncompounded):** -15.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 1 | 6 | 0 | -2.15% | -15.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 1 | 6 | 0 | -2.15% | -15.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 1 | 14.3% | 1 | 6 | 0 | -2.15% | -15.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 1363.90 | 1258.51 | 1258.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 14:15:00 | 1386.30 | 1277.55 | 1269.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 1318.40 | 1331.59 | 1303.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:45:00 | 1320.00 | 1331.59 | 1303.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1340.90 | 1371.28 | 1339.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 1358.60 | 1371.28 | 1339.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 1360.00 | 1371.14 | 1339.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1366.10 | 1370.57 | 1340.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:00:00 | 1358.50 | 1374.20 | 1344.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1397.50 | 1373.35 | 1347.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1416.60 | 1374.99 | 1349.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:45:00 | 1413.30 | 1409.57 | 1376.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1320.80 | 1401.55 | 1375.09 | SL hit (close<static) qty=1.00 sl=1330.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1320.80 | 1401.55 | 1375.09 | SL hit (close<static) qty=1.00 sl=1330.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1320.80 | 1401.55 | 1375.09 | SL hit (close<static) qty=1.00 sl=1330.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1320.80 | 1401.55 | 1375.09 | SL hit (close<static) qty=1.00 sl=1330.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1320.80 | 1401.55 | 1375.09 | SL hit (close<static) qty=1.00 sl=1340.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1320.80 | 1401.55 | 1375.09 | SL hit (close<static) qty=1.00 sl=1340.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1550.00 | 1394.89 | 1373.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-20 10:15:00 | 1705.00 | 1574.85 | 1542.83 | Target hit (10%) qty=1.00 alert=retest2 |
| CROSSOVER_SKIP | 2026-01-19 10:15:00 | 1478.30 | 1607.64 | 1607.90 | min_gap filter: gap=0.018% < 0.030% |
| TREND_RESET | 2026-01-19 10:15:00 | 1478.30 | 1607.64 | 1607.90 | EMA inversion without crossover edge (EMA200=1607.64 EMA400=1607.90) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-31 10:15:00 | 1358.60 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-31 10:45:00 | 1360.00 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-08-01 09:15:00 | 1366.10 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-08-05 12:00:00 | 1358.50 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1416.60 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2025-08-25 14:45:00 | 1413.30 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -6.54% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1550.00 | 2025-11-20 10:15:00 | 1705.00 | TARGET_HIT | 1.00 | 10.00% |
