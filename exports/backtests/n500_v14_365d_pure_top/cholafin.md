# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1671.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 0
- **Target hits / Stop hits / Partials:** 0 / 8 / 8
- **Avg / median % per leg:** 3.58% / 4.43%
- **Sum % (uncompounded):** 57.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 16 | 100.0% | 0 | 8 | 8 | 3.58% | 57.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 16 | 100.0% | 0 | 8 | 8 | 3.58% | 57.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 16 | 100.0% | 0 | 8 | 8 | 3.58% | 57.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 1490.50 | 1549.86 | 1550.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 11:15:00 | 1476.80 | 1546.04 | 1548.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1527.40 | 1499.79 | 1519.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1527.40 | 1499.79 | 1519.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1527.40 | 1499.79 | 1519.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1527.40 | 1499.79 | 1519.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1523.50 | 1500.03 | 1519.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 1520.00 | 1500.69 | 1519.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 1520.00 | 1500.90 | 1519.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 1519.80 | 1501.62 | 1519.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1512.30 | 1502.48 | 1519.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1527.90 | 1503.00 | 1519.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:45:00 | 1521.30 | 1504.01 | 1519.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 1524.90 | 1505.73 | 1519.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1522.30 | 1506.20 | 1519.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1524.60 | 1506.78 | 1519.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1444.00 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1444.00 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1443.81 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1445.23 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1448.65 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1446.18 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1448.37 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 1436.68 | 1501.74 | 1516.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1503.50 | 1483.69 | 1503.07 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1622.60 | 1515.28 | 1514.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1633.10 | 1517.52 | 1516.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1681.40 | 1682.10 | 1636.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1657.10 | 1692.39 | 1658.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1657.10 | 1692.39 | 1658.29 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1611.30 | 1682.53 | 1682.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1569.70 | 1680.14 | 1681.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1687.10 | 1672.27 | 1677.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1687.10 | 1672.27 | 1677.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1687.10 | 1672.27 | 1677.47 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1719.30 | 1682.30 | 1682.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1742.80 | 1682.90 | 1682.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 1686.50 | 1697.82 | 1690.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 1686.50 | 1697.82 | 1690.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1686.50 | 1697.82 | 1690.92 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 1546.00 | 1689.08 | 1689.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1512.00 | 1672.92 | 1681.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1527.00 | 1506.87 | 1575.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 1569.10 | 1511.50 | 1573.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1569.10 | 1511.50 | 1573.22 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-18 13:30:00 | 1520.00 | 2025-08-28 09:15:00 | 1444.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 15:15:00 | 1520.00 | 2025-08-28 09:15:00 | 1444.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-19 12:00:00 | 1519.80 | 2025-08-28 09:15:00 | 1443.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1512.30 | 2025-08-28 09:15:00 | 1445.23 | PARTIAL | 0.50 | 4.43% |
| SELL | retest2 | 2025-08-20 14:45:00 | 1521.30 | 2025-08-28 09:15:00 | 1448.65 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2025-08-22 11:30:00 | 1524.90 | 2025-08-28 09:15:00 | 1446.18 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1522.30 | 2025-08-28 09:15:00 | 1448.37 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1524.60 | 2025-08-28 14:15:00 | 1436.68 | PARTIAL | 0.50 | 5.77% |
| SELL | retest2 | 2025-08-18 13:30:00 | 1520.00 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-08-18 15:15:00 | 1520.00 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-08-19 12:00:00 | 1519.80 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1512.30 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2025-08-20 14:45:00 | 1521.30 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2025-08-22 11:30:00 | 1524.90 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1522.30 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1524.60 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.42% |
