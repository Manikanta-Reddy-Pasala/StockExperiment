# Torrent Power Ltd. (TORNTPOWER)

## Backtest Summary

- **Window:** 2025-04-21 09:15:00 → 2026-05-08 15:15:00 (1822 bars)
- **Last close:** 1717.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 0
- **Avg / median % per leg:** 0.77% / -1.87%
- **Sum % (uncompounded):** 9.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.58% | 20.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.58% | 20.6% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.85% | -11.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.85% | -11.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 3 | 25.0% | 3 | 9 | 0 | 0.77% | 9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 1315.20 | 1301.09 | 1301.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 12:15:00 | 1318.00 | 1302.24 | 1301.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 10:15:00 | 1302.20 | 1302.62 | 1301.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 1302.20 | 1302.62 | 1301.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1302.20 | 1302.62 | 1301.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 1302.20 | 1302.62 | 1301.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1307.50 | 1302.67 | 1301.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 1312.00 | 1302.75 | 1301.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 15:15:00 | 1300.60 | 1302.81 | 1301.95 | SL hit (close<static) qty=1.00 sl=1301.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1265.70 | 1300.94 | 1301.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1259.10 | 1300.53 | 1300.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1293.00 | 1292.59 | 1296.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 14:00:00 | 1293.00 | 1292.59 | 1296.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1291.90 | 1292.32 | 1296.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 1295.00 | 1292.32 | 1296.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1295.60 | 1292.10 | 1295.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 1295.60 | 1292.10 | 1295.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1281.40 | 1292.00 | 1295.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1264.60 | 1291.37 | 1295.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 1304.20 | 1287.82 | 1292.62 | SL hit (close>static) qty=1.00 sl=1303.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1273.20 | 1287.76 | 1292.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:45:00 | 1273.20 | 1287.39 | 1292.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:15:00 | 1272.90 | 1287.39 | 1292.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1290.00 | 1286.64 | 1291.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1291.80 | 1286.64 | 1291.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1296.00 | 1286.73 | 1291.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 1294.00 | 1286.73 | 1291.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1308.20 | 1286.94 | 1291.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1308.20 | 1286.94 | 1291.69 | SL hit (close>static) qty=1.00 sl=1303.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1308.20 | 1286.94 | 1291.69 | SL hit (close>static) qty=1.00 sl=1303.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1308.20 | 1286.94 | 1291.69 | SL hit (close>static) qty=1.00 sl=1303.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1308.20 | 1286.94 | 1291.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 1400.60 | 1296.22 | 1296.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 1402.70 | 1299.32 | 1297.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 15:15:00 | 1320.10 | 1320.54 | 1309.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:15:00 | 1309.90 | 1320.54 | 1309.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1329.10 | 1320.62 | 1309.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 1315.80 | 1320.62 | 1309.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1313.50 | 1328.63 | 1316.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1313.80 | 1328.63 | 1316.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1314.30 | 1328.49 | 1316.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 1322.00 | 1326.25 | 1315.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 1297.60 | 1325.52 | 1315.50 | SL hit (close<static) qty=1.00 sl=1304.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 1334.90 | 1321.94 | 1314.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:45:00 | 1322.50 | 1329.62 | 1319.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1297.80 | 1329.31 | 1319.07 | SL hit (close<static) qty=1.00 sl=1304.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1297.80 | 1329.31 | 1319.07 | SL hit (close<static) qty=1.00 sl=1304.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 1322.00 | 1329.31 | 1319.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1295.40 | 1328.97 | 1318.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 1295.40 | 1328.97 | 1318.95 | SL hit (close<static) qty=1.00 sl=1304.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 1313.00 | 1328.97 | 1318.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 15:15:00 | 1444.30 | 1350.12 | 1332.16 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:00:00 | 1306.40 | 1423.44 | 1412.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 1303.80 | 1422.23 | 1412.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 1437.04 | 1414.21 | 1408.99 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-08 09:15:00 | 1434.18 | 1414.21 | 1408.99 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-03 12:45:00 | 1312.00 | 2025-12-03 15:15:00 | 1300.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1264.60 | 2025-12-26 12:15:00 | 1304.20 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1273.20 | 2025-12-31 12:15:00 | 1308.20 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-12-30 11:45:00 | 1273.20 | 2025-12-31 12:15:00 | 1308.20 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-12-30 12:15:00 | 1272.90 | 2025-12-31 12:15:00 | 1308.20 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-01-22 15:00:00 | 1322.00 | 2026-01-23 11:15:00 | 1297.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-01-28 09:15:00 | 1334.90 | 2026-02-01 14:15:00 | 1297.80 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-02-01 13:45:00 | 1322.50 | 2026-02-01 14:15:00 | 1297.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-02-01 14:30:00 | 1322.00 | 2026-02-01 15:15:00 | 1295.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-02-02 09:15:00 | 1313.00 | 2026-02-09 15:15:00 | 1444.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:00:00 | 1306.40 | 2026-04-08 09:15:00 | 1437.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:15:00 | 1303.80 | 2026-04-08 09:15:00 | 1434.18 | TARGET_HIT | 1.00 | 10.00% |
