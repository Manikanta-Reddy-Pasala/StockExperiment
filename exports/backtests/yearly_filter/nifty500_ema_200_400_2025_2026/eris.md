# Eris Lifesciences Ltd. (ERIS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1389.70
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
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 6
- **Target hits / Stop hits / Partials:** 0 / 11 / 5
- **Avg / median % per leg:** 0.41% / 0.57%
- **Sum % (uncompounded):** 6.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 10 | 62.5% | 0 | 11 | 5 | 0.41% | 6.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 10 | 62.5% | 0 | 11 | 5 | 0.41% | 6.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 10 | 62.5% | 0 | 11 | 5 | 0.41% | 6.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1647.10 | 1697.67 | 1697.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1591.70 | 1692.80 | 1695.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1627.00 | 1620.33 | 1646.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1627.00 | 1620.33 | 1646.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1616.50 | 1619.78 | 1645.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:45:00 | 1608.20 | 1619.84 | 1643.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:15:00 | 1609.00 | 1619.81 | 1643.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 1607.60 | 1618.21 | 1641.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:45:00 | 1607.10 | 1617.45 | 1640.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1594.20 | 1612.65 | 1635.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:00:00 | 1584.20 | 1612.13 | 1635.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1584.30 | 1610.77 | 1633.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1579.00 | 1609.31 | 1632.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 1584.20 | 1608.41 | 1631.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1527.79 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1528.55 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1527.22 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1526.74 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 12:15:00 | 1599.10 | 1592.77 | 1619.27 | SL hit (close>ema200) qty=0.50 sl=1592.77 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-30 11:45:00 | 1608.20 | 2025-11-13 13:15:00 | 1527.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:15:00 | 1609.00 | 2025-11-13 13:15:00 | 1528.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 13:00:00 | 1607.60 | 2025-11-13 13:15:00 | 1527.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:45:00 | 1607.10 | 2025-11-13 13:15:00 | 1526.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 11:45:00 | 1608.20 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.57% |
| SELL | retest2 | 2025-10-31 10:15:00 | 1609.00 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2025-11-03 13:00:00 | 1607.60 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2025-11-04 09:45:00 | 1607.10 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest2 | 2025-11-10 11:00:00 | 1584.20 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1584.30 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1579.00 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2025-11-12 12:00:00 | 1584.20 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-11-28 12:15:00 | 1594.60 | 2025-12-09 12:15:00 | 1640.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-11-28 15:15:00 | 1593.00 | 2025-12-09 12:15:00 | 1640.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-12-10 15:15:00 | 1593.00 | 2025-12-26 09:15:00 | 1513.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 15:15:00 | 1593.00 | 2026-01-05 11:15:00 | 1558.00 | STOP_HIT | 0.50 | 2.20% |
