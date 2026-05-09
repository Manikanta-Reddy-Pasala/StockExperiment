# TECHM (TECHM)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1460.90
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
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -2.45% / -2.39%
- **Sum % (uncompounded):** -17.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.45% | -17.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.45% | -17.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.45% | -17.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1546.10 | 1472.93 | 1472.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.65 | 1474.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.91 | 1546.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:30:00 | 1579.90 | 1579.91 | 1546.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.68 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 1441.90 | 1589.59 | 1589.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 14:15:00 | 1439.60 | 1588.09 | 1589.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.70 | 1416.16 | 1473.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 1425.20 | 1416.16 | 1473.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.81 | 1461.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1455.80 | 1413.81 | 1461.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1451.40 | 1414.19 | 1461.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1437.50 | 1422.32 | 1461.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1440.80 | 1424.27 | 1461.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1462.10 | 1425.95 | 1459.41 | SL hit (close>static) qty=1.00 sl=1462.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1462.10 | 1425.95 | 1459.41 | SL hit (close>static) qty=1.00 sl=1462.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1437.50 | 1446.83 | 1465.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1471.80 | 1446.17 | 1464.63 | SL hit (close>static) qty=1.00 sl=1462.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:15:00 | 1446.90 | 1446.17 | 1464.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1460.10 | 1446.31 | 1464.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:30:00 | 1466.50 | 1446.31 | 1464.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 1465.00 | 1446.50 | 1464.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-22 15:15:00 | 1465.00 | 1446.50 | 1464.61 | SL hit (close>static) qty=1.00 sl=1462.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 1428.20 | 1446.50 | 1464.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1425.00 | 1446.29 | 1464.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:15:00 | 1416.90 | 1445.64 | 1463.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 1480.00 | 1437.46 | 1456.12 | SL hit (close>static) qty=1.00 sl=1473.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-24 11:45:00 | 1504.60 | 2025-12-03 10:15:00 | 1546.30 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1500.10 | 2025-12-03 10:15:00 | 1546.30 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1437.50 | 2026-04-15 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-10 09:15:00 | 1440.80 | 2026-04-15 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1437.50 | 2026-04-22 13:15:00 | 1471.80 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-04-22 14:15:00 | 1446.90 | 2026-04-22 15:15:00 | 1465.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-23 15:15:00 | 1416.90 | 2026-04-30 13:15:00 | 1480.00 | STOP_HIT | 1.00 | -4.45% |
