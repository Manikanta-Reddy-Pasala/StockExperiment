# TECHM (TECHM)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 1463.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 0 / 8 / 2
- **Avg / median % per leg:** 1.46% / -2.24%
- **Sum % (uncompounded):** 14.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 0 | 2 | 2 | 8.77% | 35.1% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 8.77% | 35.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.41% | -20.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.13% | -8.3% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.05% | -12.2% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 4.47% | 26.8% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.05% | -12.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 1522.10 | 1467.10 | 1466.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 1536.10 | 1470.07 | 1468.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.94 | 1545.72 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 1588.60 | 1580.01 | 1546.09 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1591.40 | 1580.12 | 1546.32 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 14:15:00 | 1584.70 | 1580.26 | 1547.06 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-09 15:15:00 | 1582.20 | 1580.28 | 1547.24 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-12 11:15:00 | 1583.50 | 1580.04 | 1547.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 12:15:00 | 1585.70 | 1580.10 | 1547.80 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 09:15:00 | 1830.11 | 1656.58 | 1605.07 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 09:15:00 | 1823.55 | 1656.58 | 1605.07 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1660.54 | 1608.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1628.90 | 1660.54 | 1608.86 | SL hit (close<ema200) qty=0.50 sl=1660.54 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1628.90 | 1660.54 | 1608.86 | SL hit (close<ema200) qty=0.50 sl=1660.54 alert=retest1 |

### Cycle 2 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 1440.90 | 1584.91 | 1585.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1394.00 | 1583.02 | 1584.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.90 | 1415.56 | 1471.75 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-03-25 15:15:00 | 1404.00 | 1416.27 | 1469.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1407.50 | 1416.18 | 1468.81 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 2520m) |
| Cross detected — sustain check pending | 2026-04-01 13:15:00 | 1406.30 | 1412.77 | 1462.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1404.40 | 1412.69 | 1462.19 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.44 | 1460.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1464.00 | 1416.14 | 1460.15 | SL hit (close>ema400) qty=1.00 sl=1460.15 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1464.00 | 1416.14 | 1460.15 | SL hit (close>ema400) qty=1.00 sl=1460.15 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 1437.50 | 1421.99 | 1460.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-09 10:15:00 | 1444.90 | 1422.22 | 1460.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1436.60 | 1424.08 | 1459.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1431.00 | 1424.15 | 1459.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 15:15:00 | 1438.00 | 1424.74 | 1459.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1423.30 | 1424.72 | 1459.06 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 1467.00 | 1426.43 | 1458.41 | SL hit (close>static) qty=1.00 sl=1464.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 1467.00 | 1426.43 | 1458.41 | SL hit (close>static) qty=1.00 sl=1464.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1422.60 | 1443.78 | 1463.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1410.50 | 1443.45 | 1463.02 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1472.30 | 1443.45 | 1462.83 | SL hit (close>static) qty=1.00 sl=1464.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1425.00 | 1443.65 | 1462.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1434.60 | 1443.56 | 1462.50 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1458.60 | 1432.87 | 1454.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 1466.70 | 1435.20 | 1454.52 | SL hit (close>static) qty=1.00 sl=1464.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-03 09:15:00 | 1830.11 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-03 09:15:00 | 1823.55 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-04 09:15:00 | 1628.90 | STOP_HIT | 0.50 | 2.36% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-04 09:15:00 | 1628.90 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest1 | 2026-03-27 09:15:00 | 1407.50 | 2026-04-07 09:15:00 | 1464.00 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest1 | 2026-04-01 14:15:00 | 1404.40 | 2026-04-07 09:15:00 | 1464.00 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1431.00 | 2026-04-15 11:15:00 | 1467.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1423.30 | 2026-04-15 11:15:00 | 1467.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-04-22 11:15:00 | 1410.50 | 2026-04-22 13:15:00 | 1472.30 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-04-23 10:15:00 | 1434.60 | 2026-04-30 12:15:00 | 1466.70 | STOP_HIT | 1.00 | -2.24% |
