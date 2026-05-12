# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 725.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 19 |
| TARGET_HIT | 13 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 21
- **Target hits / Stop hits / Partials:** 13 / 27 / 19
- **Avg / median % per leg:** 3.24% / 5.00%
- **Sum % (uncompounded):** 191.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.52% | -9.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.52% | -9.1% |
| SELL (all) | 53 | 38 | 71.7% | 13 | 21 | 19 | 3.78% | 200.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 38 | 71.7% | 13 | 21 | 19 | 3.78% | 200.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 59 | 38 | 64.4% | 13 | 27 | 19 | 3.24% | 191.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 13:15:00 | 1360.00 | 1509.76 | 1510.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 1349.00 | 1505.18 | 1507.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 14:15:00 | 1480.05 | 1473.59 | 1490.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 15:00:00 | 1480.05 | 1473.59 | 1490.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 1492.00 | 1473.77 | 1490.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 1477.45 | 1473.77 | 1490.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1481.35 | 1473.85 | 1490.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1466.10 | 1488.38 | 1494.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 1430.30 | 1487.67 | 1493.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 15:15:00 | 1392.79 | 1474.50 | 1486.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-29 13:15:00 | 1510.70 | 1445.38 | 1467.30 | SL hit (close>ema200) qty=0.50 sl=1445.38 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 1526.45 | 1480.37 | 1480.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 1537.30 | 1480.94 | 1480.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 12:15:00 | 1489.45 | 1492.05 | 1486.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 12:45:00 | 1489.85 | 1492.05 | 1486.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 1480.00 | 1491.93 | 1486.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 1482.75 | 1491.93 | 1486.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1475.65 | 1491.77 | 1486.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 1475.65 | 1491.77 | 1486.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1454.20 | 1489.34 | 1485.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 1454.20 | 1489.34 | 1485.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1444.60 | 1488.90 | 1485.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 1444.60 | 1488.90 | 1485.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 1448.20 | 1486.46 | 1484.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:45:00 | 1445.10 | 1486.46 | 1484.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1353.05 | 1481.59 | 1481.68 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 1510.30 | 1480.79 | 1480.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 15:15:00 | 1515.10 | 1483.02 | 1481.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1726.40 | 1750.99 | 1669.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 1726.40 | 1750.99 | 1669.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1737.15 | 1791.34 | 1739.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 1736.00 | 1791.34 | 1739.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1738.80 | 1790.82 | 1739.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 15:00:00 | 1749.70 | 1779.96 | 1737.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 1729.95 | 1778.64 | 1738.09 | SL hit (close<static) qty=1.00 sl=1735.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 1673.00 | 1723.60 | 1723.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 1659.80 | 1718.05 | 1720.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 15:15:00 | 1716.95 | 1716.85 | 1720.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 15:15:00 | 1716.95 | 1716.85 | 1720.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1716.95 | 1716.85 | 1720.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1730.75 | 1716.85 | 1720.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1727.30 | 1716.95 | 1720.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 1731.80 | 1716.95 | 1720.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1737.15 | 1717.15 | 1720.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 1737.15 | 1717.15 | 1720.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1754.25 | 1720.80 | 1721.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 1755.80 | 1720.80 | 1721.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 1788.25 | 1723.05 | 1722.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 15:15:00 | 1796.80 | 1723.78 | 1723.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 1737.30 | 1741.08 | 1732.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 12:00:00 | 1737.30 | 1741.08 | 1732.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 1731.65 | 1740.98 | 1732.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:45:00 | 1732.00 | 1740.98 | 1732.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 1729.80 | 1740.87 | 1732.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:45:00 | 1735.30 | 1740.87 | 1732.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 1720.00 | 1740.66 | 1732.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 1720.00 | 1740.66 | 1732.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1699.00 | 1740.19 | 1732.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 1699.00 | 1740.19 | 1732.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1691.65 | 1739.71 | 1732.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1688.45 | 1739.71 | 1732.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1719.20 | 1737.10 | 1731.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 1719.20 | 1737.10 | 1731.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1695.00 | 1736.68 | 1731.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 1695.00 | 1736.68 | 1731.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 1437.00 | 1725.76 | 1725.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1357.65 | 1710.09 | 1717.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1439.90 | 1436.78 | 1520.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 1519.90 | 1444.69 | 1516.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1519.90 | 1444.69 | 1516.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 1519.90 | 1444.69 | 1516.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1534.40 | 1445.59 | 1516.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:45:00 | 1541.15 | 1445.59 | 1516.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1532.80 | 1468.69 | 1519.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 1532.80 | 1468.69 | 1519.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1519.15 | 1478.24 | 1520.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 1519.15 | 1478.24 | 1520.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1526.55 | 1479.36 | 1520.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 1528.45 | 1479.36 | 1520.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1522.55 | 1479.79 | 1520.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:30:00 | 1516.55 | 1480.59 | 1520.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:45:00 | 1516.75 | 1481.35 | 1520.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1496.30 | 1481.77 | 1520.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1513.30 | 1483.69 | 1519.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 1440.72 | 1482.56 | 1518.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 1440.91 | 1482.56 | 1518.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 1437.63 | 1482.19 | 1517.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 12:15:00 | 1421.48 | 1480.56 | 1516.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 1471.75 | 1471.33 | 1506.37 | SL hit (close>ema200) qty=0.50 sl=1471.33 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 1321.50 | 1284.70 | 1284.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 1326.00 | 1286.06 | 1285.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1324.40 | 1350.04 | 1326.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1324.40 | 1350.04 | 1326.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1324.40 | 1350.04 | 1326.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1316.80 | 1350.04 | 1326.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1315.60 | 1349.70 | 1326.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:30:00 | 1315.90 | 1349.70 | 1326.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1270.50 | 1310.29 | 1310.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1263.60 | 1308.23 | 1309.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 1295.50 | 1295.30 | 1301.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:30:00 | 1295.00 | 1295.30 | 1301.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1226.50 | 1220.89 | 1242.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1226.50 | 1220.89 | 1242.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1229.80 | 1220.98 | 1242.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 1235.60 | 1220.98 | 1242.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 1236.00 | 1221.39 | 1241.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 1218.20 | 1221.39 | 1241.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 1227.40 | 1221.66 | 1241.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1244.40 | 1222.15 | 1241.28 | SL hit (close>static) qty=1.00 sl=1244.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 1259.80 | 1206.27 | 1206.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1283.20 | 1207.55 | 1206.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 1206.30 | 1216.00 | 1211.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 1206.30 | 1216.00 | 1211.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1206.30 | 1216.00 | 1211.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 1195.90 | 1216.00 | 1211.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1204.60 | 1215.89 | 1211.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 1198.40 | 1215.89 | 1211.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1218.20 | 1215.91 | 1211.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 13:15:00 | 1221.70 | 1215.91 | 1211.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1196.00 | 1215.45 | 1211.36 | SL hit (close<static) qty=1.00 sl=1202.60 alert=retest2 |

### Cycle 11 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1162.40 | 1208.37 | 1208.56 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 1230.50 | 1208.66 | 1208.65 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 1168.10 | 1208.65 | 1208.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 1160.60 | 1208.18 | 1208.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1193.60 | 1193.21 | 1200.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 10:00:00 | 1193.60 | 1193.21 | 1200.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1200.40 | 1193.20 | 1200.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 1200.40 | 1193.20 | 1200.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1209.90 | 1193.37 | 1200.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1214.30 | 1193.37 | 1200.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1208.00 | 1193.51 | 1200.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1225.00 | 1193.51 | 1200.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1188.20 | 1189.19 | 1197.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1167.10 | 1189.50 | 1196.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 1162.20 | 1189.29 | 1196.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:00:00 | 1162.60 | 1187.95 | 1195.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1108.74 | 1183.19 | 1193.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1104.09 | 1183.19 | 1193.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1104.47 | 1183.19 | 1193.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-29 13:15:00 | 1050.39 | 1157.16 | 1177.08 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 09:15:00 | 1466.10 | 2024-04-18 15:15:00 | 1392.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 09:15:00 | 1466.10 | 2024-04-29 13:15:00 | 1510.70 | STOP_HIT | 0.50 | -3.04% |
| SELL | retest2 | 2024-04-15 09:15:00 | 1430.30 | 2024-04-29 13:15:00 | 1510.70 | STOP_HIT | 1.00 | -5.62% |
| SELL | retest2 | 2024-05-10 09:15:00 | 1467.30 | 2024-05-14 15:15:00 | 1498.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-05-14 14:00:00 | 1471.90 | 2024-05-14 15:15:00 | 1498.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-05-15 09:15:00 | 1488.30 | 2024-05-16 09:15:00 | 1536.00 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-05-21 09:15:00 | 1485.40 | 2024-05-24 09:15:00 | 1526.45 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-05-21 11:30:00 | 1488.40 | 2024-05-24 09:15:00 | 1526.45 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-05-21 13:15:00 | 1485.20 | 2024-05-24 09:15:00 | 1526.45 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2024-09-09 15:00:00 | 1749.70 | 2024-09-10 11:15:00 | 1729.95 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-09-10 13:45:00 | 1743.05 | 2024-09-17 09:15:00 | 1734.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-09-17 09:30:00 | 1748.80 | 2024-09-18 11:15:00 | 1730.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-09-17 10:45:00 | 1745.95 | 2024-09-18 11:15:00 | 1730.75 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-12-18 12:30:00 | 1516.55 | 2024-12-20 15:15:00 | 1440.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 14:45:00 | 1516.75 | 2024-12-20 15:15:00 | 1440.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1496.30 | 2024-12-23 09:15:00 | 1437.63 | PARTIAL | 0.50 | 3.92% |
| SELL | retest2 | 2024-12-20 09:30:00 | 1513.30 | 2024-12-23 12:15:00 | 1421.48 | PARTIAL | 0.50 | 6.07% |
| SELL | retest2 | 2024-12-18 12:30:00 | 1516.55 | 2024-12-30 14:15:00 | 1471.75 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2024-12-18 14:45:00 | 1516.75 | 2024-12-30 14:15:00 | 1471.75 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1496.30 | 2024-12-30 14:15:00 | 1471.75 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2024-12-20 09:30:00 | 1513.30 | 2024-12-30 14:15:00 | 1471.75 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-01-31 09:15:00 | 1401.00 | 2025-02-12 09:15:00 | 1330.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 12:45:00 | 1406.80 | 2025-02-12 09:15:00 | 1336.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 1407.10 | 2025-02-12 09:15:00 | 1336.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 12:15:00 | 1409.85 | 2025-02-12 09:15:00 | 1339.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 11:30:00 | 1431.75 | 2025-02-12 09:15:00 | 1360.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 13:15:00 | 1433.35 | 2025-02-12 09:15:00 | 1361.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 14:30:00 | 1428.30 | 2025-02-12 09:15:00 | 1356.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 10:45:00 | 1430.30 | 2025-02-12 09:15:00 | 1358.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1410.50 | 2025-02-12 09:15:00 | 1339.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 09:15:00 | 1401.00 | 2025-02-24 09:15:00 | 1288.58 | TARGET_HIT | 0.50 | 8.02% |
| SELL | retest2 | 2025-01-31 12:45:00 | 1406.80 | 2025-02-24 09:15:00 | 1290.01 | TARGET_HIT | 0.50 | 8.30% |
| SELL | retest2 | 2025-02-01 11:45:00 | 1407.10 | 2025-02-24 09:15:00 | 1285.47 | TARGET_HIT | 0.50 | 8.64% |
| SELL | retest2 | 2025-02-01 12:15:00 | 1409.85 | 2025-02-24 09:15:00 | 1287.27 | TARGET_HIT | 0.50 | 8.69% |
| SELL | retest2 | 2025-02-04 11:30:00 | 1431.75 | 2025-02-24 10:15:00 | 1260.90 | TARGET_HIT | 0.50 | 11.93% |
| SELL | retest2 | 2025-02-05 13:15:00 | 1433.35 | 2025-02-24 10:15:00 | 1266.12 | TARGET_HIT | 0.50 | 11.67% |
| SELL | retest2 | 2025-02-05 14:30:00 | 1428.30 | 2025-02-24 10:15:00 | 1266.39 | TARGET_HIT | 0.50 | 11.34% |
| SELL | retest2 | 2025-02-06 10:45:00 | 1430.30 | 2025-02-24 10:15:00 | 1268.87 | TARGET_HIT | 0.50 | 11.29% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1410.50 | 2025-02-24 10:15:00 | 1269.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1218.20 | 2025-09-09 10:15:00 | 1244.40 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-09-08 15:15:00 | 1227.40 | 2025-09-09 10:15:00 | 1244.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-26 13:15:00 | 1228.40 | 2025-09-30 12:15:00 | 1166.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 13:15:00 | 1228.40 | 2025-09-30 14:15:00 | 1105.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-14 10:30:00 | 1229.60 | 2025-11-19 09:15:00 | 1215.00 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-11-18 13:45:00 | 1198.10 | 2025-11-19 09:15:00 | 1215.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-18 14:30:00 | 1199.30 | 2025-11-21 14:15:00 | 1168.12 | PARTIAL | 0.50 | 2.60% |
| SELL | retest2 | 2025-11-18 14:30:00 | 1199.30 | 2025-11-24 10:15:00 | 1199.20 | STOP_HIT | 0.50 | 0.01% |
| SELL | retest2 | 2025-11-20 14:45:00 | 1197.50 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1194.20 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1193.40 | 2025-11-27 10:15:00 | 1214.30 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-25 10:00:00 | 1194.00 | 2025-11-27 10:15:00 | 1214.30 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-12-09 13:15:00 | 1221.70 | 2025-12-10 10:15:00 | 1196.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-12-11 11:15:00 | 1225.60 | 2025-12-16 09:15:00 | 1183.10 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1167.10 | 2026-01-21 09:15:00 | 1108.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:15:00 | 1162.20 | 2026-01-21 09:15:00 | 1104.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:00:00 | 1162.60 | 2026-01-21 09:15:00 | 1104.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1167.10 | 2026-01-29 13:15:00 | 1050.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 10:15:00 | 1162.20 | 2026-01-29 14:15:00 | 1045.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 15:00:00 | 1162.60 | 2026-01-29 14:15:00 | 1046.34 | TARGET_HIT | 0.50 | 10.00% |
