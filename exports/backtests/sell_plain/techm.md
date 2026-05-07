# TECHM (TECHM)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 1450.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 7 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -3.76% / -3.69%
- **Sum % (uncompounded):** -15.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.76% | -15.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.75% | -7.5% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.77% | -7.5% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.75% | -7.5% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.77% | -7.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 1344.10 | 1577.74 | 1578.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 1341.40 | 1575.39 | 1577.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.70 | 1416.08 | 1470.38 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-25 15:15:00 | 1403.50 | 1416.73 | 1467.81 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1407.50 | 1416.64 | 1467.51 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 2520m) |
| Cross detected — sustain check pending | 2026-04-01 13:15:00 | 1406.30 | 1413.15 | 1461.29 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 15:15:00 | 1405.70 | 1412.99 | 1460.74 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.76 | 1459.25 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1459.40 | 1413.76 | 1459.25 | SL hit (close>ema400) qty=1.00 sl=1459.25 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1459.40 | 1413.76 | 1459.25 | SL hit (close>ema400) qty=1.00 sl=1459.25 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-10 10:15:00 | 1431.00 | 1424.42 | 1458.85 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-10 12:15:00 | 1437.20 | 1424.61 | 1458.60 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1423.30 | 1424.98 | 1458.11 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 1433.70 | 1425.07 | 1457.99 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1422.60 | 1446.56 | 1463.49 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1414.90 | 1445.89 | 1462.98 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1471.80 | 1446.15 | 1463.02 | SL hit (close>static) qty=1.00 sl=1464.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1425.00 | 1446.26 | 1462.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-23 10:15:00 | 1434.60 | 1446.15 | 1462.69 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 13:15:00 | 1428.70 | 1445.83 | 1462.29 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:15:00 | 1416.90 | 1445.33 | 1461.87 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 1466.70 | 1437.02 | 1454.69 | SL hit (close>static) qty=1.00 sl=1464.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-27 09:15:00 | 1407.50 | 2026-04-06 09:15:00 | 1459.40 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest1 | 2026-04-01 15:15:00 | 1405.70 | 2026-04-06 09:15:00 | 1459.40 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2026-04-22 12:15:00 | 1414.90 | 2026-04-22 13:15:00 | 1471.80 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2026-04-23 15:15:00 | 1416.90 | 2026-04-30 12:15:00 | 1466.70 | STOP_HIT | 1.00 | -3.51% |
