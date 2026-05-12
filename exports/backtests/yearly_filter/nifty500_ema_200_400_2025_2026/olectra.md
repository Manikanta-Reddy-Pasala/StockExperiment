# Olectra Greentech Ltd. (OLECTRA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1345.00
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
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 16
- **Target hits / Stop hits / Partials:** 0 / 19 / 3
- **Avg / median % per leg:** -1.79% / -2.28%
- **Sum % (uncompounded):** -39.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.82% | -19.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.82% | -19.8% |
| SELL (all) | 15 | 6 | 40.0% | 0 | 12 | 3 | -1.31% | -19.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 0 | 12 | 3 | -1.31% | -19.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 6 | 27.3% | 0 | 19 | 3 | -1.79% | -39.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1336.40 | 1222.10 | 1221.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 15:15:00 | 1350.00 | 1225.65 | 1223.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1213.00 | 1228.87 | 1225.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1213.00 | 1228.87 | 1225.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1213.00 | 1228.87 | 1225.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 1213.00 | 1228.87 | 1225.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1200.60 | 1228.59 | 1225.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1235.00 | 1226.84 | 1224.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 1217.40 | 1228.33 | 1225.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1216.00 | 1228.00 | 1225.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1172.30 | 1223.82 | 1223.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1172.30 | 1223.82 | 1223.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1158.00 | 1213.80 | 1218.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.13 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 1278.30 | 1213.39 | 1213.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 1284.20 | 1217.35 | 1215.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 1573.20 | 1578.42 | 1495.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 13:00:00 | 1573.20 | 1578.42 | 1495.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1489.00 | 1561.47 | 1509.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 1489.00 | 1561.47 | 1509.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1484.20 | 1560.70 | 1509.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 1484.00 | 1560.70 | 1509.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1516.50 | 1508.40 | 1495.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 1518.40 | 1508.48 | 1495.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 1526.00 | 1508.48 | 1495.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:45:00 | 1517.80 | 1508.79 | 1496.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 1519.10 | 1508.61 | 1496.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1492.20 | 1508.42 | 1496.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1492.20 | 1508.42 | 1496.66 | SL hit (close<static) qty=1.00 sl=1494.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 1412.40 | 1486.59 | 1486.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1391.10 | 1481.20 | 1484.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 1261.00 | 1257.13 | 1329.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:30:00 | 1267.70 | 1257.13 | 1329.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1060.25 | 974.94 | 1048.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 1060.25 | 974.94 | 1048.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1033.15 | 975.52 | 1047.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1027.50 | 975.52 | 1047.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1018.15 | 978.30 | 1046.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 1075.50 | 979.27 | 1046.71 | SL hit (close>static) qty=1.00 sl=1069.45 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 12:15:00 | 1227.05 | 1071.79 | 1071.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 1232.60 | 1095.51 | 1084.09 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-30 09:15:00 | 1235.00 | 2025-06-16 09:15:00 | 1172.30 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest2 | 2025-06-11 14:15:00 | 1217.40 | 2025-06-16 09:15:00 | 1172.30 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-06-12 09:30:00 | 1216.00 | 2025-06-16 09:15:00 | 1172.30 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-11-06 11:30:00 | 1518.40 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-11-06 12:15:00 | 1526.00 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-11-06 14:45:00 | 1517.80 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-11-07 15:00:00 | 1519.10 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1027.50 | 2026-03-20 10:15:00 | 1075.50 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-03-20 09:30:00 | 1018.15 | 2026-03-20 10:15:00 | 1075.50 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2026-03-23 12:15:00 | 1028.05 | 2026-03-24 09:15:00 | 1071.00 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2026-03-23 12:45:00 | 1021.20 | 2026-03-24 09:15:00 | 1071.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-03-27 09:45:00 | 1038.25 | 2026-03-30 09:15:00 | 986.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:45:00 | 1038.25 | 2026-04-01 09:15:00 | 1033.00 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2026-04-01 09:45:00 | 1036.35 | 2026-04-02 09:15:00 | 987.52 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1039.50 | 2026-04-02 09:15:00 | 987.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 09:45:00 | 1036.35 | 2026-04-02 11:15:00 | 1003.50 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1039.50 | 2026-04-02 11:15:00 | 1003.50 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2026-04-01 13:15:00 | 1039.50 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1027.10 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2026-04-06 14:15:00 | 1026.95 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-07 09:30:00 | 1025.35 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-07 15:00:00 | 1026.75 | 2026-04-09 09:15:00 | 1119.55 | STOP_HIT | 1.00 | -9.04% |
