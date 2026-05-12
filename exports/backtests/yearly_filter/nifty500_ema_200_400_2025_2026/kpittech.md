# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 725.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 9
- **Target hits / Stop hits / Partials:** 4 / 11 / 5
- **Avg / median % per leg:** 2.41% / 1.19%
- **Sum % (uncompounded):** 48.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.79% | -5.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.79% | -5.6% |
| SELL (all) | 18 | 11 | 61.1% | 4 | 9 | 5 | 2.99% | 53.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 11 | 61.1% | 4 | 9 | 5 | 2.99% | 53.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 11 | 55.0% | 4 | 11 | 5 | 2.41% | 48.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 1321.50 | 1284.70 | 1284.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 1326.00 | 1286.06 | 1285.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1324.40 | 1350.04 | 1326.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1324.40 | 1350.04 | 1326.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1324.40 | 1350.04 | 1326.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1316.80 | 1350.04 | 1326.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1315.60 | 1349.70 | 1326.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:30:00 | 1315.90 | 1349.70 | 1326.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1270.50 | 1310.29 | 1310.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1263.60 | 1308.23 | 1309.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 1295.50 | 1295.30 | 1301.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:30:00 | 1295.00 | 1295.30 | 1301.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1226.50 | 1220.89 | 1242.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1226.50 | 1220.89 | 1242.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1229.80 | 1220.98 | 1242.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 1235.60 | 1220.98 | 1242.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 1236.00 | 1221.39 | 1241.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 1218.20 | 1221.39 | 1241.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 1227.40 | 1221.66 | 1241.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1244.40 | 1222.15 | 1241.28 | SL hit (close>static) qty=1.00 sl=1244.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-03 14:15:00)

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

### Cycle 4 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1162.40 | 1208.37 | 1208.56 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 1230.50 | 1208.66 | 1208.65 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-12-30 13:15:00)

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
