# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1669.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 33 |
| PARTIAL | 7 |
| TARGET_HIT | 8 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 22
- **Target hits / Stop hits / Partials:** 8 / 26 / 7
- **Avg / median % per leg:** 1.70% / -0.83%
- **Sum % (uncompounded):** 69.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 6 | 12 | 0 | 1.26% | 22.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 6 | 33.3% | 6 | 12 | 0 | 1.26% | 22.6% |
| SELL (all) | 23 | 13 | 56.5% | 2 | 14 | 7 | 2.05% | 47.2% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.43% | 2.9% |
| SELL @ 3rd Alert (retest2) | 21 | 12 | 57.1% | 2 | 13 | 6 | 2.11% | 44.4% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.43% | 2.9% |
| retest2 (combined) | 39 | 18 | 46.2% | 8 | 25 | 6 | 1.72% | 67.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 15:15:00 | 1189.97 | 1240.80 | 1241.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 1171.00 | 1240.10 | 1240.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 15:15:00 | 1232.50 | 1231.33 | 1235.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 15:15:00 | 1232.50 | 1231.33 | 1235.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 1232.50 | 1231.33 | 1235.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 1234.53 | 1220.86 | 1226.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 1225.22 | 1221.32 | 1226.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:30:00 | 1225.00 | 1221.32 | 1226.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 1216.85 | 1221.28 | 1226.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 15:00:00 | 1216.85 | 1221.28 | 1226.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 1229.00 | 1220.94 | 1225.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 12:45:00 | 1225.97 | 1220.94 | 1225.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 1230.03 | 1221.03 | 1225.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 1227.50 | 1221.22 | 1225.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 10:15:00 | 1237.63 | 1221.49 | 1225.94 | SL hit (close>static) qty=1.00 sl=1235.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1229.15 | 1183.51 | 1183.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1258.95 | 1186.06 | 1184.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 14:15:00 | 1218.22 | 1221.46 | 1206.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 14:30:00 | 1213.72 | 1221.46 | 1206.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1216.70 | 1221.30 | 1206.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:45:00 | 1220.08 | 1221.26 | 1206.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 1189.63 | 1219.73 | 1206.52 | SL hit (close<static) qty=1.00 sl=1190.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1488.25 | 1654.69 | 1654.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1460.38 | 1652.75 | 1653.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1622.45 | 1615.09 | 1632.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:30:00 | 1623.98 | 1615.09 | 1632.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 1647.00 | 1614.36 | 1631.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:45:00 | 1647.10 | 1614.36 | 1631.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1654.80 | 1614.76 | 1631.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 1654.80 | 1614.76 | 1631.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1626.18 | 1615.27 | 1631.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:00:00 | 1617.88 | 1615.30 | 1631.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1619.83 | 1614.90 | 1630.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1536.99 | 1610.29 | 1627.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1538.84 | 1610.29 | 1627.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 1589.38 | 1575.80 | 1603.18 | SL hit (close>ema200) qty=0.50 sl=1575.80 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 1701.00 | 1394.37 | 1393.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1736.40 | 1468.64 | 1433.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1687.50 | 1691.99 | 1594.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1687.50 | 1691.99 | 1594.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2040.25 | 2117.78 | 2025.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 2073.00 | 2065.87 | 2021.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 2020.95 | 2064.92 | 2022.17 | SL hit (close<static) qty=1.00 sl=2021.05 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 1999.05 | 2254.17 | 2254.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 1958.75 | 2246.06 | 2250.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 1555.10 | 1553.74 | 1710.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 1499.60 | 1563.54 | 1694.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1424.62 | 1531.65 | 1653.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1531.80 | 1511.86 | 1629.14 | SL hit (close>ema200) qty=0.50 sl=1511.86 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-16 09:15:00 | 1227.50 | 2024-04-16 10:15:00 | 1237.63 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-04-16 12:00:00 | 1227.90 | 2024-04-22 14:15:00 | 1166.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-16 13:00:00 | 1227.55 | 2024-04-22 14:15:00 | 1166.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-16 12:00:00 | 1227.90 | 2024-04-24 13:15:00 | 1217.47 | STOP_HIT | 0.50 | 0.85% |
| SELL | retest2 | 2024-04-16 13:00:00 | 1227.55 | 2024-04-24 13:15:00 | 1217.47 | STOP_HIT | 0.50 | 0.82% |
| SELL | retest2 | 2024-04-16 15:15:00 | 1217.45 | 2024-04-25 10:15:00 | 1228.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-04-19 09:15:00 | 1205.00 | 2024-04-29 09:15:00 | 1233.75 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-04-26 10:30:00 | 1220.00 | 2024-04-29 09:15:00 | 1233.75 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-04-26 12:15:00 | 1221.43 | 2024-04-29 12:15:00 | 1237.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-05-02 15:00:00 | 1213.13 | 2024-05-03 09:15:00 | 1236.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-05-03 11:30:00 | 1222.58 | 2024-05-07 12:15:00 | 1161.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 13:15:00 | 1219.63 | 2024-05-07 13:15:00 | 1158.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:30:00 | 1222.58 | 2024-05-13 12:15:00 | 1100.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-03 13:15:00 | 1219.63 | 2024-05-13 12:15:00 | 1097.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-21 11:15:00 | 1218.03 | 2024-07-03 09:15:00 | 1229.15 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-07-02 10:00:00 | 1224.90 | 2024-07-03 09:15:00 | 1229.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-07-22 10:45:00 | 1220.08 | 2024-07-23 14:15:00 | 1189.63 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-08-01 12:15:00 | 1224.75 | 2024-08-05 10:15:00 | 1175.13 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-08-01 15:00:00 | 1221.63 | 2024-08-05 10:15:00 | 1175.13 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2024-08-02 14:15:00 | 1222.00 | 2024-08-05 10:15:00 | 1175.13 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2024-08-06 09:15:00 | 1227.90 | 2024-08-14 10:15:00 | 1197.18 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-08-16 10:00:00 | 1211.88 | 2024-08-19 11:15:00 | 1333.07 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-06 12:00:00 | 1617.88 | 2025-02-11 09:15:00 | 1536.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 11:15:00 | 1619.83 | 2025-02-11 09:15:00 | 1538.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 12:00:00 | 1617.88 | 2025-02-21 09:15:00 | 1589.38 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-02-07 11:15:00 | 1619.83 | 2025-02-21 09:15:00 | 1589.38 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-05-19 09:45:00 | 1618.65 | 2025-05-19 11:15:00 | 1639.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-13 13:45:00 | 2073.00 | 2025-10-14 13:15:00 | 2020.95 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-10-15 11:00:00 | 2074.50 | 2025-10-27 09:15:00 | 2281.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-17 09:30:00 | 2078.95 | 2025-10-27 09:15:00 | 2286.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2133.10 | 2025-10-27 09:15:00 | 2346.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-12 09:15:00 | 2237.90 | 2026-01-06 09:15:00 | 2461.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 09:15:00 | 2246.70 | 2026-01-06 09:15:00 | 2471.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-22 09:45:00 | 2237.00 | 2026-01-22 11:15:00 | 2199.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-01-22 10:30:00 | 2225.70 | 2026-01-22 11:15:00 | 2199.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-04 12:30:00 | 2261.00 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2252.50 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-02-04 14:30:00 | 2251.85 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2026-02-04 15:00:00 | 2274.40 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest1 | 2026-04-22 10:45:00 | 1499.60 | 2026-04-29 15:15:00 | 1424.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 10:45:00 | 1499.60 | 2026-05-06 11:15:00 | 1531.80 | STOP_HIT | 0.50 | -2.15% |
