# Bharat Dynamics Ltd. (BDL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1447.20
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
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 1
- **Target hits / Stop hits / Partials:** 0 / 7 / 6
- **Avg / median % per leg:** 2.80% / 3.62%
- **Sum % (uncompounded):** 36.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.92% | -0.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.92% | -0.9% |
| SELL (all) | 12 | 12 | 100.0% | 0 | 6 | 6 | 3.11% | 37.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 12 | 100.0% | 0 | 6 | 6 | 3.11% | 37.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 12 | 92.3% | 0 | 7 | 6 | 2.80% | 36.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 1590.40 | 1741.57 | 1741.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 1574.50 | 1739.91 | 1740.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 1570.40 | 1527.44 | 1593.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 1570.40 | 1527.44 | 1593.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1585.80 | 1529.84 | 1592.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 1588.40 | 1529.84 | 1592.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1608.40 | 1532.12 | 1592.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 1608.40 | 1532.12 | 1592.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1600.00 | 1532.79 | 1592.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1604.50 | 1532.79 | 1592.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1614.10 | 1534.40 | 1592.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 1619.50 | 1534.40 | 1592.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 1604.50 | 1536.42 | 1593.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 1604.50 | 1536.42 | 1593.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1583.80 | 1558.99 | 1596.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:45:00 | 1576.00 | 1559.20 | 1596.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 1580.70 | 1559.61 | 1596.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1579.00 | 1560.30 | 1596.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 1579.20 | 1560.49 | 1596.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1582.60 | 1561.08 | 1595.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 1599.00 | 1561.08 | 1595.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1598.00 | 1561.45 | 1595.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1598.00 | 1561.45 | 1595.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1584.60 | 1561.68 | 1595.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 1582.70 | 1561.87 | 1595.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1497.20 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1501.66 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1500.05 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1500.24 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1503.57 | 1559.35 | 1592.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1567.90 | 1549.23 | 1583.71 | SL hit (close>ema200) qty=0.50 sl=1549.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1567.90 | 1549.23 | 1583.71 | SL hit (close>ema200) qty=0.50 sl=1549.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1567.90 | 1549.23 | 1583.71 | SL hit (close>ema200) qty=0.50 sl=1549.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1567.90 | 1549.23 | 1583.71 | SL hit (close>ema200) qty=0.50 sl=1549.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1567.90 | 1549.23 | 1583.71 | SL hit (close>ema200) qty=0.50 sl=1549.23 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1581.20 | 1530.66 | 1546.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 1502.14 | 1533.92 | 1546.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 1524.00 | 1521.76 | 1537.43 | SL hit (close>ema200) qty=0.50 sl=1521.76 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 1356.50 | 1333.43 | 1333.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 1373.00 | 1334.10 | 1333.67 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-17 09:15:00 | 1807.50 | 2025-07-17 09:15:00 | 1790.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-23 11:45:00 | 1576.00 | 2025-09-26 14:15:00 | 1497.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 1580.70 | 2025-09-26 14:15:00 | 1501.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1579.00 | 2025-09-26 14:15:00 | 1500.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:00:00 | 1579.20 | 2025-09-26 14:15:00 | 1500.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-09-26 14:15:00 | 1503.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:45:00 | 1576.00 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2025-09-23 13:45:00 | 1580.70 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.81% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1579.00 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.70% |
| SELL | retest2 | 2025-09-24 10:00:00 | 1579.20 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-10-03 09:15:00 | 1567.90 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1581.20 | 2025-11-24 09:15:00 | 1502.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1581.20 | 2025-12-01 09:15:00 | 1524.00 | STOP_HIT | 0.50 | 3.62% |
