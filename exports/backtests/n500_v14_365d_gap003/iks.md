# Inventurus Knowledge Solutions Ltd. (IKS)

## Backtest Summary

- **Window:** 2024-12-19 09:15:00 → 2026-05-08 15:15:00 (2389 bars)
- **Last close:** 1686.00
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
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -5.79% / -4.91%
- **Sum % (uncompounded):** -28.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -5.79% | -29.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -5.79% | -29.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -5.79% | -29.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1516.30 | 1641.31 | 1641.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 1480.50 | 1639.71 | 1641.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 1392.20 | 1391.33 | 1465.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 09:30:00 | 1383.90 | 1391.33 | 1465.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1456.10 | 1393.19 | 1463.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:00:00 | 1428.60 | 1436.28 | 1471.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 09:45:00 | 1429.50 | 1436.14 | 1470.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1416.80 | 1436.38 | 1470.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:00:00 | 1425.00 | 1436.15 | 1469.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 1458.00 | 1435.95 | 1467.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1593.60 | 1452.94 | 1472.15 | SL hit (close>static) qty=1.00 sl=1573.80 alert=retest2 |
| CROSSOVER_SKIP | 2026-05-05 14:15:00 | 1696.10 | 1489.92 | 1489.71 | min_gap filter: gap=0.012% < 0.030% |
| TREND_RESET | 2026-05-05 14:15:00 | 1696.10 | 1489.92 | 1489.71 | EMA inversion without crossover edge (EMA200=1489.92 EMA400=1489.71) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-20 15:00:00 | 1428.60 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2026-04-21 09:45:00 | 1429.50 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1416.80 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2026-04-22 11:00:00 | 1425.00 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2026-04-24 10:30:00 | 1458.00 | 2026-04-30 09:15:00 | 1593.60 | STOP_HIT | 1.00 | -9.30% |
