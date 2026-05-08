# TECHM (TECHM)

## Backtest Summary

- **Window:** 2025-11-10 09:15:00 → 2026-05-08 15:15:00 (854 bars)
- **Last close:** 1460.90
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
| ALERT3 | 1 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -1.91% / -2.52%
- **Sum % (uncompounded):** -11.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -1.91% | -11.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.80% | -7.6% |
| SELL @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.97% | -3.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.80% | -7.6% |
| retest2 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.97% | -3.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 1357.30 | 1562.71 | 1563.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 15:15:00 | 1356.00 | 1556.72 | 1560.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.70 | 1415.98 | 1466.57 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-03-25 15:15:00 | 1403.50 | 1416.64 | 1464.21 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1407.50 | 1416.55 | 1463.93 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 2520m) |
| Cross detected — sustain check pending | 2026-04-01 13:15:00 | 1406.30 | 1413.07 | 1458.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1404.40 | 1412.99 | 1457.75 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.69 | 1456.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1459.40 | 1413.69 | 1456.13 | SL hit (close>ema400) qty=1.00 sl=1456.13 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1459.40 | 1413.69 | 1456.13 | SL hit (close>ema400) qty=1.00 sl=1456.13 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-10 10:15:00 | 1431.00 | 1424.37 | 1456.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 11:15:00 | 1430.90 | 1424.43 | 1456.03 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1423.30 | 1424.93 | 1455.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 1433.70 | 1425.02 | 1455.39 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 1467.00 | 1426.64 | 1455.02 | SL hit (close>static) qty=1.00 sl=1464.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1422.60 | 1446.53 | 1461.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1410.50 | 1446.17 | 1461.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1471.80 | 1446.12 | 1460.95 | SL hit (close>static) qty=1.00 sl=1464.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1425.00 | 1446.23 | 1460.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-23 10:15:00 | 1434.60 | 1446.12 | 1460.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 13:15:00 | 1428.70 | 1445.80 | 1460.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 1424.10 | 1445.59 | 1460.10 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 12:15:00 | 1352.89 | 1442.21 | 1458.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1452.70 | 1434.59 | 1452.65 | SL hit (close>ema200) qty=0.50 sl=1434.59 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-27 09:15:00 | 1407.50 | 2026-04-06 09:15:00 | 1459.40 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest1 | 2026-04-01 14:15:00 | 1404.40 | 2026-04-06 09:15:00 | 1459.40 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2026-04-10 11:15:00 | 1430.90 | 2026-04-15 11:15:00 | 1467.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-04-22 11:15:00 | 1410.50 | 2026-04-22 13:15:00 | 1471.80 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2026-04-23 14:15:00 | 1424.10 | 2026-04-24 12:15:00 | 1352.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 14:15:00 | 1424.10 | 2026-04-29 09:15:00 | 1452.70 | STOP_HIT | 0.50 | -2.01% |
