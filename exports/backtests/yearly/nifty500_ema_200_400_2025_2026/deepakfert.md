# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1342.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 4
- **Target hits / Stop hits / Partials:** 1 / 9 / 6
- **Avg / median % per leg:** 2.84% / 3.71%
- **Sum % (uncompounded):** 45.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 12 | 75.0% | 1 | 9 | 6 | 2.84% | 45.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 12 | 75.0% | 1 | 9 | 6 | 2.84% | 45.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 12 | 75.0% | 1 | 9 | 6 | 2.84% | 45.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 1458.00 | 1492.39 | 1492.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 1428.00 | 1488.39 | 1490.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 1471.40 | 1456.58 | 1471.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 11:15:00 | 1471.40 | 1456.58 | 1471.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1471.40 | 1456.58 | 1471.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:30:00 | 1475.50 | 1456.58 | 1471.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1477.00 | 1456.78 | 1471.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 1481.10 | 1456.78 | 1471.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 1474.10 | 1456.96 | 1471.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:15:00 | 1460.30 | 1456.96 | 1471.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 1493.80 | 1456.30 | 1469.45 | SL hit (close>static) qty=1.00 sl=1478.70 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 1496.00 | 1480.70 | 1480.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 1511.80 | 1481.01 | 1480.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 10:15:00 | 1496.80 | 1500.22 | 1491.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 10:15:00 | 1496.80 | 1500.22 | 1491.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1496.80 | 1500.22 | 1491.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 1500.30 | 1500.22 | 1491.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1493.00 | 1500.17 | 1491.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1479.80 | 1500.17 | 1491.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1485.00 | 1500.02 | 1491.76 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 1438.50 | 1484.48 | 1484.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1431.20 | 1483.95 | 1484.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1477.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1477.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1477.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 1531.60 | 1471.77 | 1477.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1522.00 | 1472.27 | 1477.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:30:00 | 1507.50 | 1475.06 | 1479.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 1501.00 | 1475.80 | 1479.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1505.50 | 1477.88 | 1480.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 1507.80 | 1478.65 | 1480.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1494.40 | 1479.04 | 1480.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 1492.60 | 1479.04 | 1480.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1493.40 | 1479.18 | 1480.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 1493.20 | 1479.18 | 1480.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:30:00 | 1492.90 | 1479.45 | 1480.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1512.90 | 1479.92 | 1481.15 | SL hit (close>static) qty=1.00 sl=1502.40 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1267.75 | 1073.09 | 1072.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 1270.05 | 1075.05 | 1073.21 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-17 14:15:00 | 1460.30 | 2025-09-23 12:15:00 | 1493.80 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-10-29 10:30:00 | 1507.50 | 2025-11-04 09:15:00 | 1512.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-29 14:15:00 | 1501.00 | 2025-11-04 09:15:00 | 1512.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-31 11:15:00 | 1505.50 | 2025-11-04 14:15:00 | 1504.60 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-10-31 14:00:00 | 1507.80 | 2025-11-06 09:15:00 | 1432.12 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-11-03 11:15:00 | 1493.20 | 2025-11-06 09:15:00 | 1425.95 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1492.90 | 2025-11-06 09:15:00 | 1430.22 | PARTIAL | 0.50 | 4.20% |
| SELL | retest2 | 2025-11-04 13:45:00 | 1492.60 | 2025-11-06 09:15:00 | 1432.41 | PARTIAL | 0.50 | 4.03% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1438.80 | 2025-11-11 10:15:00 | 1366.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 14:00:00 | 1507.80 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-11-03 11:15:00 | 1493.20 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1492.90 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-11-04 13:45:00 | 1492.60 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1438.80 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | -0.91% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1432.10 | 2025-11-24 15:15:00 | 1360.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1432.10 | 2025-12-08 09:15:00 | 1288.89 | TARGET_HIT | 0.50 | 10.00% |
