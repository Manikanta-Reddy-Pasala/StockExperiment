# TECHM (TECHM)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 1463.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 10 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 3
- **Avg / median % per leg:** 1.41% / -1.28%
- **Sum % (uncompounded):** 15.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -2.07% | -14.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.13% | -8.3% |
| SELL @ 3rd Alert (retest2) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.25% | -6.2% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 3.62% | 21.7% |
| retest2 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.25% | -6.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 1568.80 | 1478.45 | 1478.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 1572.20 | 1483.47 | 1480.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.94 | 1547.99 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 1588.60 | 1580.01 | 1548.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1591.40 | 1580.12 | 1548.56 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 14:15:00 | 1584.70 | 1580.27 | 1549.25 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-09 15:15:00 | 1582.20 | 1580.29 | 1549.42 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-12 11:15:00 | 1583.50 | 1580.05 | 1549.76 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 12:15:00 | 1585.70 | 1580.10 | 1549.94 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 12:15:00 | 1664.98 | 1585.74 | 1555.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 1670.97 | 1587.33 | 1556.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-27 14:15:00 | 1744.27 | 1626.43 | 1584.25 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-28 09:15:00 | 1750.54 | 1628.83 | 1585.87 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1660.54 | 1610.15 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 1439.60 | 1586.36 | 1586.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1394.00 | 1583.02 | 1585.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.90 | 1415.56 | 1472.15 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-03-25 15:15:00 | 1404.00 | 1416.27 | 1469.49 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1407.50 | 1416.18 | 1469.18 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 2520m) |
| Cross detected — sustain check pending | 2026-04-01 13:15:00 | 1406.30 | 1412.77 | 1462.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1404.40 | 1412.69 | 1462.54 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.44 | 1460.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1464.00 | 1416.14 | 1460.46 | SL hit (close>ema400) qty=1.00 sl=1460.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1464.00 | 1416.14 | 1460.46 | SL hit (close>ema400) qty=1.00 sl=1460.46 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 1437.50 | 1421.99 | 1460.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-09 10:15:00 | 1444.90 | 1422.22 | 1460.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1436.60 | 1424.08 | 1460.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1431.00 | 1424.15 | 1460.10 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 15:15:00 | 1438.00 | 1424.74 | 1459.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1423.30 | 1424.72 | 1459.33 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 1467.00 | 1426.43 | 1458.67 | SL hit (close>static) qty=1.00 sl=1464.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 1467.00 | 1426.43 | 1458.67 | SL hit (close>static) qty=1.00 sl=1464.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1422.60 | 1443.78 | 1463.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1410.50 | 1443.45 | 1463.25 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1472.30 | 1443.45 | 1463.05 | SL hit (close>static) qty=1.00 sl=1464.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1425.00 | 1443.65 | 1462.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1434.60 | 1443.56 | 1462.72 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 1362.87 | 1440.78 | 1460.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1452.90 | 1432.61 | 1454.44 | SL hit (close>ema200) qty=0.50 sl=1432.61 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1458.60 | 1432.87 | 1454.46 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-01-16 12:15:00 | 1664.98 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-01-16 14:15:00 | 1670.97 | PARTIAL | 0.50 | 5.38% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-01-27 14:15:00 | 1744.27 | TARGET_HIT | 0.50 | 9.61% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-01-28 09:15:00 | 1750.54 | TARGET_HIT | 0.50 | 10.40% |
| SELL | retest1 | 2026-03-27 09:15:00 | 1407.50 | 2026-04-07 09:15:00 | 1464.00 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest1 | 2026-04-01 14:15:00 | 1404.40 | 2026-04-07 09:15:00 | 1464.00 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1431.00 | 2026-04-15 11:15:00 | 1467.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1423.30 | 2026-04-15 11:15:00 | 1467.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-04-22 11:15:00 | 1410.50 | 2026-04-22 13:15:00 | 1472.30 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-04-23 10:15:00 | 1434.60 | 2026-04-24 11:15:00 | 1362.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 10:15:00 | 1434.60 | 2026-04-29 09:15:00 | 1452.90 | STOP_HIT | 0.50 | -1.28% |
