# TBO Tek Ltd. (TBOTEK)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
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
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 16 / 4
- **Target hits / Stop hits / Partials:** 4 / 8 / 8
- **Avg / median % per leg:** 4.45% / 5.00%
- **Sum % (uncompounded):** 88.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 16 | 80.0% | 4 | 8 | 8 | 4.45% | 88.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 16 | 80.0% | 4 | 8 | 8 | 4.45% | 88.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 16 | 80.0% | 4 | 8 | 8 | 4.45% | 88.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1465.20 | 1606.22 | 1606.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 14:15:00 | 1454.30 | 1588.18 | 1597.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1495.00 | 1494.71 | 1536.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:30:00 | 1493.70 | 1494.71 | 1536.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1559.00 | 1496.64 | 1536.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 1559.00 | 1496.64 | 1536.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1572.80 | 1497.40 | 1536.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 1572.80 | 1497.40 | 1536.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1543.90 | 1504.23 | 1538.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 1543.90 | 1504.23 | 1538.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1539.00 | 1504.58 | 1538.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1448.20 | 1504.58 | 1538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1482.80 | 1504.36 | 1537.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1424.00 | 1502.57 | 1536.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 1444.50 | 1499.71 | 1533.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 14:15:00 | 1442.70 | 1498.03 | 1531.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 15:15:00 | 1434.00 | 1497.53 | 1531.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:15:00 | 1372.27 | 1484.71 | 1522.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 1370.57 | 1479.33 | 1518.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1352.80 | 1476.98 | 1516.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1362.30 | 1476.98 | 1516.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 1300.05 | 1459.71 | 1505.23 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-13 09:15:00 | 1424.00 | 2026-02-19 09:15:00 | 1372.27 | PARTIAL | 0.50 | 3.63% |
| SELL | retest2 | 2026-02-16 09:45:00 | 1444.50 | 2026-02-19 14:15:00 | 1370.57 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2026-02-16 14:15:00 | 1442.70 | 2026-02-20 09:15:00 | 1352.80 | PARTIAL | 0.50 | 6.23% |
| SELL | retest2 | 2026-02-16 15:15:00 | 1434.00 | 2026-02-20 09:15:00 | 1362.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1424.00 | 2026-02-24 09:15:00 | 1300.05 | TARGET_HIT | 0.50 | 8.70% |
| SELL | retest2 | 2026-02-16 09:45:00 | 1444.50 | 2026-02-25 09:15:00 | 1298.43 | TARGET_HIT | 0.50 | 10.11% |
| SELL | retest2 | 2026-02-16 14:15:00 | 1442.70 | 2026-02-25 10:15:00 | 1281.60 | TARGET_HIT | 0.50 | 11.17% |
| SELL | retest2 | 2026-02-16 15:15:00 | 1434.00 | 2026-02-25 10:15:00 | 1290.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-17 13:30:00 | 1290.10 | 2026-04-24 09:15:00 | 1225.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 13:30:00 | 1290.10 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2026-04-17 15:00:00 | 1291.80 | 2026-04-24 09:15:00 | 1227.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 15:00:00 | 1291.80 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2026-04-20 12:00:00 | 1289.70 | 2026-04-24 09:15:00 | 1225.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 12:00:00 | 1289.70 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2026-04-21 10:15:00 | 1292.10 | 2026-04-24 09:15:00 | 1227.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 10:15:00 | 1292.10 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2026-04-28 11:30:00 | 1258.50 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-04-30 10:00:00 | 1255.00 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-04-30 11:45:00 | 1257.00 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1259.30 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -1.87% |
