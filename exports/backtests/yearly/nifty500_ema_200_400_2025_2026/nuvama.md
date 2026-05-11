# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1631.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 14
- **Target hits / Stop hits / Partials:** 0 / 16 / 0
- **Avg / median % per leg:** -2.16% / -1.07%
- **Sum % (uncompounded):** -34.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 2 | 12.5% | 0 | 16 | 0 | -2.16% | -34.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 0 | 16 | 0 | -2.16% | -34.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 2 | 12.5% | 0 | 16 | 0 | -2.16% | -34.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1382.00 | 1421.65 | 1421.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 1374.70 | 1418.24 | 1419.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 1294.00 | 1290.89 | 1331.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 14:00:00 | 1294.00 | 1290.89 | 1331.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1355.80 | 1291.68 | 1331.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 1356.60 | 1291.68 | 1331.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1329.60 | 1292.06 | 1331.19 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1422.40 | 1357.21 | 1357.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 1450.00 | 1364.88 | 1361.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 1438.90 | 1440.45 | 1414.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 12:00:00 | 1438.90 | 1440.45 | 1414.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1412.10 | 1439.76 | 1414.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:45:00 | 1407.70 | 1439.76 | 1414.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1410.00 | 1439.46 | 1414.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1448.90 | 1436.67 | 1413.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 11:45:00 | 1413.20 | 1446.11 | 1423.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:00:00 | 1415.30 | 1445.51 | 1423.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:45:00 | 1415.60 | 1445.20 | 1423.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1422.90 | 1443.81 | 1423.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 13:30:00 | 1427.00 | 1443.79 | 1423.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 15:00:00 | 1427.00 | 1443.36 | 1424.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1409.70 | 1442.86 | 1423.93 | SL hit (close<static) qty=1.00 sl=1419.50 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1270.00 | 1434.29 | 1434.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1241.40 | 1429.18 | 1432.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 14:15:00 | 1392.00 | 1387.16 | 1407.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 1392.00 | 1387.16 | 1407.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1378.20 | 1384.15 | 1403.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 1378.20 | 1384.15 | 1403.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1256.80 | 1200.95 | 1257.07 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1360.60 | 1290.46 | 1290.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1362.70 | 1291.84 | 1291.12 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-30 12:45:00 | 1476.00 | 2025-08-01 09:15:00 | 1409.70 | STOP_HIT | 1.00 | -4.49% |
| BUY | retest2 | 2025-07-31 11:00:00 | 1459.10 | 2025-08-01 09:15:00 | 1409.70 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-07-31 14:00:00 | 1465.60 | 2025-08-01 09:15:00 | 1409.70 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2025-11-26 09:15:00 | 1448.90 | 2025-12-09 09:15:00 | 1409.70 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-12-04 11:45:00 | 1413.20 | 2025-12-09 09:15:00 | 1409.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-12-04 14:00:00 | 1415.30 | 2025-12-19 11:15:00 | 1418.40 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-12-04 14:45:00 | 1415.60 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-12-05 13:30:00 | 1427.00 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-08 15:00:00 | 1427.00 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1431.30 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-19 14:30:00 | 1426.20 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-12-23 11:00:00 | 1442.20 | 2026-01-20 10:15:00 | 1433.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-23 12:30:00 | 1441.70 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-12-29 10:30:00 | 1452.50 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2026-01-09 09:30:00 | 1440.00 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2026-01-16 09:15:00 | 1487.20 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -6.13% |
