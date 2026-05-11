# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1554.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 0 / 12 / 3
- **Avg / median % per leg:** -0.51% / -1.81%
- **Sum % (uncompounded):** -7.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.82% | -5.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.82% | -5.5% |
| SELL (all) | 12 | 6 | 50.0% | 0 | 9 | 3 | -0.19% | -2.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 6 | 50.0% | 0 | 9 | 3 | -0.19% | -2.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 6 | 40.0% | 0 | 12 | 3 | -0.51% | -7.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1482.20 | 1419.69 | 1419.39 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 1376.50 | 1419.35 | 1419.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 15:15:00 | 1368.80 | 1412.30 | 1415.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1415.00 | 1407.52 | 1413.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1407.90 | 1407.52 | 1413.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 1401.90 | 1407.48 | 1413.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:45:00 | 1403.30 | 1407.44 | 1412.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1331.81 | 1393.86 | 1404.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1333.13 | 1393.86 | 1404.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 1379.30 | 1375.44 | 1391.97 | SL hit (close>ema200) qty=0.50 sl=1375.44 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1444.70 | 1402.50 | 1402.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1450.60 | 1402.98 | 1402.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1444.20 | 1454.87 | 1434.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:45:00 | 1442.00 | 1454.87 | 1434.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1434.20 | 1454.66 | 1434.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 1423.70 | 1454.66 | 1434.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1432.10 | 1454.44 | 1434.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1444.50 | 1454.21 | 1434.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:00:00 | 1440.20 | 1454.07 | 1434.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 14:45:00 | 1442.10 | 1453.85 | 1434.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 1416.00 | 1452.65 | 1434.59 | SL hit (close<static) qty=1.00 sl=1427.20 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1371.70 | 1420.89 | 1421.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1357.20 | 1411.30 | 1415.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 1417.80 | 1408.85 | 1414.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1403.60 | 1408.80 | 1414.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 1392.30 | 1408.62 | 1414.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1380.70 | 1408.21 | 1413.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 1435.00 | 1407.12 | 1413.24 | SL hit (close>static) qty=1.00 sl=1418.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1441.10 | 1348.96 | 1348.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 1448.00 | 1352.50 | 1350.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 1399.00 | 1414.53 | 1391.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 13:15:00 | 1391.70 | 1414.30 | 1391.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1391.70 | 1414.30 | 1391.28 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 11:30:00 | 1400.40 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-05-15 12:30:00 | 1396.70 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-05-15 13:15:00 | 1399.30 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-05-16 10:00:00 | 1397.20 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1401.90 | 2025-06-19 12:15:00 | 1331.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 12:45:00 | 1403.30 | 2025-06-19 12:15:00 | 1333.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1401.90 | 2025-06-30 13:15:00 | 1379.30 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-06-12 12:45:00 | 1403.30 | 2025-06-30 13:15:00 | 1379.30 | STOP_HIT | 0.50 | 1.71% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1444.50 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-08-04 13:00:00 | 1440.20 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-08-04 14:45:00 | 1442.10 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-08-25 11:30:00 | 1392.30 | 2025-08-26 14:15:00 | 1435.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1380.70 | 2025-08-26 14:15:00 | 1435.00 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-08-29 10:15:00 | 1388.50 | 2025-09-08 13:15:00 | 1319.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 10:15:00 | 1388.50 | 2025-09-19 13:15:00 | 1361.30 | STOP_HIT | 0.50 | 1.96% |
