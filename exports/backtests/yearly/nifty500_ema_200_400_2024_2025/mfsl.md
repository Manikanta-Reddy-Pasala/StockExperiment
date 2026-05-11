# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 5 |
| TARGET_HIT | 9 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 29
- **Target hits / Stop hits / Partials:** 9 / 34 / 5
- **Avg / median % per leg:** 1.21% / -1.13%
- **Sum % (uncompounded):** 58.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 7 | 11 | 0 | 2.90% | 52.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 7 | 11 | 0 | 2.90% | 52.2% |
| SELL (all) | 30 | 11 | 36.7% | 2 | 23 | 5 | 0.20% | 5.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 11 | 36.7% | 2 | 23 | 5 | 0.20% | 5.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 19 | 39.6% | 9 | 34 | 5 | 1.21% | 58.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 925.00 | 983.97 | 984.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 912.00 | 983.26 | 983.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-11 12:15:00 | 963.35 | 963.14 | 972.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:30:00 | 963.90 | 963.14 | 972.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 990.00 | 963.01 | 971.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 990.00 | 963.01 | 971.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 989.00 | 963.27 | 971.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 980.65 | 968.26 | 973.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 11:00:00 | 980.75 | 968.39 | 973.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 12:15:00 | 1002.00 | 970.25 | 974.16 | SL hit (close>static) qty=1.00 sl=994.40 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 1009.55 | 977.04 | 976.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 1017.60 | 981.94 | 979.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 1057.80 | 1058.99 | 1032.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:00:00 | 1057.80 | 1058.99 | 1032.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1031.20 | 1058.64 | 1032.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:15:00 | 1039.50 | 1058.18 | 1032.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 11:45:00 | 1038.85 | 1047.66 | 1030.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 12:15:00 | 1041.80 | 1047.66 | 1030.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:45:00 | 1039.70 | 1053.55 | 1036.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1040.00 | 1053.42 | 1036.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:15:00 | 1043.20 | 1053.42 | 1036.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:45:00 | 1052.40 | 1053.42 | 1037.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-04 09:15:00 | 1143.45 | 1062.25 | 1043.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 1143.20 | 1169.12 | 1169.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 1141.90 | 1168.84 | 1169.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 14:15:00 | 1084.85 | 1083.01 | 1109.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-30 15:00:00 | 1084.85 | 1083.01 | 1109.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 1110.95 | 1083.52 | 1108.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 1110.95 | 1083.52 | 1108.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 1110.35 | 1083.79 | 1108.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:30:00 | 1111.25 | 1083.79 | 1108.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 1109.75 | 1084.05 | 1108.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 12:45:00 | 1078.35 | 1086.20 | 1109.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 1113.80 | 1086.47 | 1109.26 | SL hit (close>static) qty=1.00 sl=1112.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 1154.95 | 1085.91 | 1085.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 10:15:00 | 1164.00 | 1099.19 | 1092.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 13:15:00 | 1558.30 | 1558.50 | 1472.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:00:00 | 1558.30 | 1558.50 | 1472.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1496.90 | 1548.83 | 1497.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 1497.50 | 1548.83 | 1497.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1484.40 | 1548.19 | 1497.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 1484.20 | 1548.19 | 1497.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1490.00 | 1545.17 | 1497.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 1510.00 | 1525.18 | 1495.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 1661.00 | 1540.50 | 1508.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1550.00 | 1561.90 | 1561.93 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 1583.10 | 1562.02 | 1561.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 1593.50 | 1562.76 | 1562.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 1672.20 | 1673.54 | 1641.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:45:00 | 1674.80 | 1673.54 | 1641.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1650.80 | 1675.26 | 1650.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1650.80 | 1675.26 | 1650.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1639.90 | 1674.90 | 1650.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1653.70 | 1674.90 | 1650.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1653.00 | 1674.67 | 1650.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 1629.90 | 1673.84 | 1650.14 | SL hit (close<static) qty=1.00 sl=1633.30 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 1597.00 | 1649.33 | 1649.41 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 1708.40 | 1649.17 | 1649.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1721.00 | 1657.05 | 1653.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1753.20 | 1771.98 | 1726.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 1753.20 | 1771.98 | 1726.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1729.20 | 1770.76 | 1726.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:30:00 | 1731.40 | 1770.76 | 1726.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1720.00 | 1769.17 | 1727.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 1720.00 | 1769.17 | 1727.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1729.10 | 1768.77 | 1727.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 13:15:00 | 1734.30 | 1768.77 | 1727.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:30:00 | 1734.30 | 1768.14 | 1727.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 1714.90 | 1766.93 | 1727.27 | SL hit (close<static) qty=1.00 sl=1719.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1560.30 | 1704.12 | 1704.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 10:15:00 | 1550.00 | 1701.31 | 1702.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 1627.50 | 1618.76 | 1653.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 11:15:00 | 1654.10 | 1619.37 | 1653.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 1654.10 | 1619.37 | 1653.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 1654.10 | 1619.37 | 1653.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 1649.80 | 1619.67 | 1653.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:30:00 | 1650.10 | 1619.67 | 1653.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1661.40 | 1620.09 | 1653.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 1661.40 | 1620.09 | 1653.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1653.50 | 1620.42 | 1653.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 1650.60 | 1620.42 | 1653.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1699.90 | 1622.95 | 1653.71 | SL hit (close>static) qty=1.00 sl=1662.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-23 15:00:00 | 959.50 | 2024-05-30 10:15:00 | 930.05 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2024-06-19 10:15:00 | 980.65 | 2024-06-20 12:15:00 | 1002.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-06-19 11:00:00 | 980.75 | 2024-06-20 12:15:00 | 1002.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-06-24 09:15:00 | 979.70 | 2024-07-01 09:15:00 | 980.30 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-06-24 10:00:00 | 980.70 | 2024-07-01 09:15:00 | 980.30 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-06-26 13:45:00 | 973.80 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-06-27 10:30:00 | 975.75 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-06-27 12:00:00 | 973.35 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-06-28 10:30:00 | 976.10 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-06-28 12:30:00 | 971.00 | 2024-07-01 13:15:00 | 998.35 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-06-28 14:30:00 | 970.80 | 2024-07-01 13:15:00 | 998.35 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-07-02 09:45:00 | 971.10 | 2024-07-02 10:15:00 | 982.85 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-07-02 12:45:00 | 971.00 | 2024-07-02 13:15:00 | 978.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-08-14 12:15:00 | 1039.50 | 2024-09-04 09:15:00 | 1143.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-21 11:45:00 | 1038.85 | 2024-09-04 09:15:00 | 1142.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-21 12:15:00 | 1041.80 | 2024-09-04 09:15:00 | 1145.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-29 14:45:00 | 1039.70 | 2024-09-04 09:15:00 | 1143.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 09:15:00 | 1043.20 | 2024-09-04 09:15:00 | 1147.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 09:45:00 | 1052.40 | 2024-09-12 09:15:00 | 1157.64 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-01 12:45:00 | 1078.35 | 2025-02-01 13:15:00 | 1113.80 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-02-03 09:30:00 | 1103.10 | 2025-02-05 09:15:00 | 1134.05 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-02-03 11:15:00 | 1103.00 | 2025-02-05 09:15:00 | 1134.05 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-02-04 09:45:00 | 1099.85 | 2025-02-05 09:15:00 | 1134.05 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-02-10 11:45:00 | 1094.90 | 2025-02-12 09:15:00 | 1040.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 13:45:00 | 1095.50 | 2025-02-12 09:15:00 | 1040.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:45:00 | 1094.90 | 2025-02-12 11:15:00 | 1093.80 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2025-02-10 13:45:00 | 1095.50 | 2025-02-12 11:15:00 | 1093.80 | STOP_HIT | 0.50 | 0.16% |
| SELL | retest2 | 2025-02-12 11:30:00 | 1089.95 | 2025-02-17 09:15:00 | 1035.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:30:00 | 1094.95 | 2025-02-17 09:15:00 | 1040.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 11:30:00 | 1089.95 | 2025-03-03 09:15:00 | 980.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-12 13:30:00 | 1094.95 | 2025-03-03 09:15:00 | 985.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-17 11:00:00 | 1041.00 | 2025-03-19 09:15:00 | 1087.25 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest2 | 2025-08-07 15:15:00 | 1510.00 | 2025-08-18 09:15:00 | 1661.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1513.80 | 2025-11-03 10:15:00 | 1550.00 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2025-12-30 09:15:00 | 1653.70 | 2025-12-30 11:15:00 | 1629.90 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1653.00 | 2025-12-30 11:15:00 | 1629.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-12-31 10:15:00 | 1654.80 | 2026-01-14 09:15:00 | 1630.90 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-01-13 10:45:00 | 1655.10 | 2026-01-14 09:15:00 | 1630.90 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-03-05 13:15:00 | 1734.30 | 2026-03-06 10:15:00 | 1714.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-03-05 14:30:00 | 1734.30 | 2026-03-06 10:15:00 | 1714.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-03-10 13:30:00 | 1736.80 | 2026-03-12 09:15:00 | 1686.00 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-03-10 14:45:00 | 1730.40 | 2026-03-12 09:15:00 | 1686.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-03-11 10:15:00 | 1749.90 | 2026-03-12 09:15:00 | 1686.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-04-10 15:15:00 | 1650.60 | 2026-04-15 09:15:00 | 1699.90 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-04-21 13:15:00 | 1651.70 | 2026-04-24 10:15:00 | 1569.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 13:15:00 | 1651.70 | 2026-05-06 09:15:00 | 1635.80 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2026-04-21 14:00:00 | 1648.00 | 2026-05-07 10:15:00 | 1679.60 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1648.50 | 2026-05-07 10:15:00 | 1679.60 | STOP_HIT | 1.00 | -1.89% |
