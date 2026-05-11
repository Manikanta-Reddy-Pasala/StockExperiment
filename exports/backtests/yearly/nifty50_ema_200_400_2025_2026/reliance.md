# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
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
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 8 / 7
- **Target hits / Stop hits / Partials:** 3 / 8 / 4
- **Avg / median % per leg:** 3.08% / 1.60%
- **Sum % (uncompounded):** 46.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 8 | 53.3% | 3 | 8 | 4 | 3.08% | 46.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 8 | 53.3% | 3 | 8 | 4 | 3.08% | 46.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 8 | 53.3% | 3 | 8 | 4 | 3.08% | 46.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1387.40 | 1415.18 | 1415.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 1378.10 | 1414.53 | 1414.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1414.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1414.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1414.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 1415.70 | 1413.28 | 1414.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1415.10 | 1413.29 | 1414.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:30:00 | 1417.50 | 1413.29 | 1414.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1419.50 | 1413.35 | 1414.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1419.50 | 1413.35 | 1414.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1422.90 | 1413.59 | 1414.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 1419.70 | 1413.67 | 1414.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 13:15:00 | 1419.10 | 1413.73 | 1414.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:15:00 | 1417.20 | 1413.80 | 1414.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 1429.90 | 1413.93 | 1414.55 | SL hit (close>static) qty=1.00 sl=1424.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1448.00 | 1397.32 | 1397.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 1455.90 | 1398.41 | 1397.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.17 | 1511.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 1539.30 | 1541.17 | 1511.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.51 | 1520.60 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.20 | 1501.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1370.70 | 1457.67 | 1476.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1449.75 | 1471.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:00:00 | 1459.10 | 1449.75 | 1471.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 1462.00 | 1450.57 | 1468.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1462.60 | 1451.01 | 1468.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 1462.20 | 1452.02 | 1468.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1388.90 | 1435.62 | 1453.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.47 | 1435.62 | 1453.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.09 | 1435.62 | 1453.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1315.80 | 1427.64 | 1448.30 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-20 11:30:00 | 1419.70 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-08-20 13:15:00 | 1419.10 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-20 14:15:00 | 1417.20 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1416.60 | 2025-09-01 09:15:00 | 1345.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1416.60 | 2025-09-12 10:15:00 | 1394.00 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2025-08-26 09:30:00 | 1404.70 | 2025-09-18 09:15:00 | 1417.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-19 09:45:00 | 1406.00 | 2025-10-17 10:15:00 | 1416.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1406.30 | 2025-10-17 10:15:00 | 1416.20 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-22 10:45:00 | 1407.00 | 2025-10-17 10:15:00 | 1416.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-02-27 09:15:00 | 1388.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-02-27 09:15:00 | 1389.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-02-27 09:15:00 | 1389.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-03-04 09:15:00 | 1315.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-03-04 09:15:00 | 1316.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-03-04 09:15:00 | 1315.98 | TARGET_HIT | 0.50 | 10.00% |
