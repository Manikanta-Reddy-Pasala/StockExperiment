# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 949.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 45 |
| ALERT2 | 44 |
| ALERT2_SKIP | 21 |
| ALERT3 | 127 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 54 |
| PARTIAL | 11 |
| TARGET_HIT | 7 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 35
- **Target hits / Stop hits / Partials:** 7 / 48 / 11
- **Avg / median % per leg:** 1.53% / -0.89%
- **Sum % (uncompounded):** 101.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 6 | 22.2% | 5 | 22 | 0 | 0.62% | 16.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 6 | 22.2% | 5 | 22 | 0 | 0.62% | 16.8% |
| SELL (all) | 39 | 25 | 64.1% | 2 | 26 | 11 | 2.16% | 84.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| SELL @ 3rd Alert (retest2) | 38 | 25 | 65.8% | 2 | 25 | 11 | 2.25% | 85.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| retest2 (combined) | 65 | 31 | 47.7% | 7 | 47 | 11 | 1.57% | 102.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1227.00 | 1182.46 | 1178.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1257.80 | 1229.11 | 1221.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 10:15:00 | 1269.90 | 1271.39 | 1259.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 10:30:00 | 1269.10 | 1271.39 | 1259.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1258.30 | 1267.68 | 1259.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 1256.70 | 1267.68 | 1259.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1251.30 | 1264.41 | 1258.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1251.30 | 1264.41 | 1258.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1243.00 | 1260.13 | 1257.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:15:00 | 1254.00 | 1260.13 | 1257.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1251.80 | 1259.06 | 1257.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 1251.80 | 1259.06 | 1257.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1257.20 | 1258.69 | 1257.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:30:00 | 1260.70 | 1258.97 | 1257.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:45:00 | 1260.60 | 1260.36 | 1258.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:30:00 | 1267.30 | 1271.54 | 1266.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-26 14:15:00 | 1386.77 | 1329.78 | 1305.78 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-26 14:15:00 | 1386.66 | 1329.78 | 1305.78 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1276.00 | 1301.86 | 1303.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1276.00 | 1301.86 | 1303.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 1267.20 | 1294.93 | 1299.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1254.00 | 1253.53 | 1265.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1254.00 | 1253.53 | 1265.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1254.00 | 1253.53 | 1265.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1259.60 | 1253.53 | 1265.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1249.50 | 1250.69 | 1259.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 1260.00 | 1250.69 | 1259.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1266.90 | 1253.58 | 1259.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 1265.10 | 1253.58 | 1259.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1259.10 | 1254.69 | 1259.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:00:00 | 1256.10 | 1254.97 | 1258.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:00:00 | 1257.50 | 1255.48 | 1258.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1269.00 | 1260.47 | 1260.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1269.00 | 1260.47 | 1260.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1269.00 | 1260.47 | 1260.36 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 1252.80 | 1259.48 | 1260.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 1250.00 | 1255.19 | 1257.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 1265.60 | 1256.78 | 1257.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 1265.60 | 1256.78 | 1257.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1265.60 | 1256.78 | 1257.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1265.60 | 1256.78 | 1257.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1270.00 | 1259.42 | 1258.78 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1255.90 | 1259.05 | 1259.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1253.20 | 1257.88 | 1258.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 1264.60 | 1251.30 | 1253.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1264.60 | 1251.30 | 1253.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1264.60 | 1251.30 | 1253.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1264.60 | 1251.30 | 1253.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1261.10 | 1253.26 | 1254.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 1258.60 | 1253.26 | 1254.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 14:15:00 | 1263.00 | 1256.41 | 1255.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 1263.00 | 1256.41 | 1255.56 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 1251.90 | 1256.62 | 1256.66 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 1260.80 | 1257.20 | 1256.90 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1253.20 | 1256.97 | 1256.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1242.90 | 1250.37 | 1253.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 1201.20 | 1195.24 | 1208.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 1201.20 | 1195.24 | 1208.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1201.20 | 1195.24 | 1208.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 1204.50 | 1195.24 | 1208.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1202.80 | 1196.75 | 1208.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 1206.20 | 1196.75 | 1208.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1208.60 | 1195.48 | 1202.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 1208.60 | 1195.48 | 1202.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1199.90 | 1196.36 | 1201.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:00:00 | 1196.70 | 1196.43 | 1201.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 1195.30 | 1196.66 | 1200.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 1195.20 | 1196.66 | 1200.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 1195.00 | 1195.31 | 1199.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1177.40 | 1178.18 | 1183.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1177.40 | 1178.18 | 1183.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1180.00 | 1178.54 | 1183.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1168.70 | 1178.54 | 1183.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1207.50 | 1177.95 | 1179.12 | SL hit (close>static) qty=1.00 sl=1184.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1208.30 | 1184.02 | 1181.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1208.30 | 1184.02 | 1181.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1208.30 | 1184.02 | 1181.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1208.30 | 1184.02 | 1181.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1208.30 | 1184.02 | 1181.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1224.10 | 1202.85 | 1193.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1221.20 | 1221.26 | 1215.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1221.20 | 1221.26 | 1215.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1212.00 | 1219.16 | 1215.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1212.00 | 1219.16 | 1215.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1214.10 | 1218.15 | 1215.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1229.00 | 1218.15 | 1215.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 1218.80 | 1222.16 | 1218.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 1218.80 | 1222.16 | 1218.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 1212.40 | 1220.21 | 1217.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 1212.40 | 1220.21 | 1217.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1221.20 | 1220.41 | 1218.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:30:00 | 1222.70 | 1218.50 | 1217.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 1222.20 | 1218.64 | 1217.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 1206.10 | 1215.92 | 1216.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 1206.10 | 1215.92 | 1216.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 1206.10 | 1215.92 | 1216.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1200.90 | 1211.17 | 1214.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 1186.30 | 1185.84 | 1191.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 1186.30 | 1185.84 | 1191.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1190.90 | 1187.04 | 1191.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1192.00 | 1187.04 | 1191.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1190.40 | 1187.71 | 1191.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1189.80 | 1187.71 | 1191.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1192.90 | 1188.46 | 1190.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1192.90 | 1188.46 | 1190.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1191.60 | 1189.09 | 1190.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1208.30 | 1189.09 | 1190.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1208.00 | 1192.87 | 1192.12 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1194.20 | 1200.35 | 1200.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 1182.40 | 1195.03 | 1197.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 1141.00 | 1139.77 | 1146.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 1146.30 | 1139.77 | 1146.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1147.10 | 1141.24 | 1146.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1147.90 | 1141.24 | 1146.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1143.90 | 1141.77 | 1146.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1140.70 | 1141.50 | 1145.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 1141.90 | 1141.58 | 1144.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 1141.70 | 1143.83 | 1145.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:00:00 | 1139.60 | 1142.98 | 1144.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1139.50 | 1140.10 | 1142.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 1139.50 | 1140.10 | 1142.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1144.90 | 1141.06 | 1142.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 1143.80 | 1141.06 | 1142.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1137.00 | 1140.25 | 1142.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:45:00 | 1134.20 | 1139.16 | 1141.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1083.66 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1084.81 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1084.62 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1082.62 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 1100.00 | 1098.75 | 1107.36 | SL hit (close>ema200) qty=0.50 sl=1098.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 1100.00 | 1098.75 | 1107.36 | SL hit (close>ema200) qty=0.50 sl=1098.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 1100.00 | 1098.75 | 1107.36 | SL hit (close>ema200) qty=0.50 sl=1098.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 1100.00 | 1098.75 | 1107.36 | SL hit (close>ema200) qty=0.50 sl=1098.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1123.00 | 1111.95 | 1111.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1123.00 | 1111.95 | 1111.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 1131.00 | 1115.76 | 1112.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1116.60 | 1127.05 | 1121.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1116.60 | 1127.05 | 1121.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1116.60 | 1127.05 | 1121.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1130.00 | 1126.86 | 1121.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 1129.20 | 1127.33 | 1122.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:30:00 | 1131.00 | 1128.34 | 1123.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 1132.60 | 1129.20 | 1124.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1127.60 | 1129.44 | 1125.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:15:00 | 1125.70 | 1129.44 | 1125.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1121.80 | 1127.91 | 1125.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1121.80 | 1127.91 | 1125.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1120.50 | 1126.43 | 1124.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1120.80 | 1126.43 | 1124.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 1118.30 | 1123.60 | 1123.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 1118.30 | 1123.60 | 1123.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 1118.30 | 1123.60 | 1123.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 1118.30 | 1123.60 | 1123.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 1118.30 | 1123.60 | 1123.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1114.90 | 1121.86 | 1122.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1101.90 | 1083.77 | 1088.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 1101.90 | 1083.77 | 1088.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1101.90 | 1083.77 | 1088.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 1114.80 | 1083.77 | 1088.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1108.00 | 1088.62 | 1090.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1105.50 | 1088.62 | 1090.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1091.90 | 1090.14 | 1090.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 1083.20 | 1089.93 | 1090.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1029.04 | 1071.87 | 1081.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-12 09:15:00 | 974.88 | 1013.39 | 1042.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 970.30 | 955.79 | 955.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 985.60 | 964.92 | 959.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 989.50 | 995.67 | 986.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 992.70 | 995.67 | 986.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 990.10 | 996.36 | 991.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 990.10 | 996.36 | 991.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 984.00 | 993.88 | 990.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 984.00 | 993.88 | 990.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 983.00 | 988.07 | 988.67 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 1040.50 | 997.83 | 992.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 1068.80 | 1012.02 | 999.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 1056.10 | 1056.68 | 1034.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 11:00:00 | 1056.10 | 1056.68 | 1034.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1045.00 | 1051.85 | 1040.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 1036.10 | 1051.85 | 1040.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1079.70 | 1057.42 | 1044.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 1095.90 | 1057.42 | 1044.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:30:00 | 1086.00 | 1065.61 | 1050.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 1091.20 | 1070.86 | 1064.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 14:45:00 | 1082.60 | 1078.61 | 1071.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1087.50 | 1080.88 | 1073.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 1097.40 | 1086.25 | 1078.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:45:00 | 1099.70 | 1088.40 | 1079.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 1097.40 | 1088.40 | 1079.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 1097.20 | 1090.79 | 1083.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1083.30 | 1089.29 | 1083.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1083.30 | 1089.29 | 1083.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1073.80 | 1086.19 | 1082.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1073.80 | 1086.19 | 1082.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1074.50 | 1083.86 | 1081.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 1072.90 | 1083.86 | 1081.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1072.80 | 1080.56 | 1080.52 | SL hit (close<static) qty=1.00 sl=1073.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1072.80 | 1080.56 | 1080.52 | SL hit (close<static) qty=1.00 sl=1073.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1072.80 | 1080.56 | 1080.52 | SL hit (close<static) qty=1.00 sl=1073.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1072.80 | 1080.56 | 1080.52 | SL hit (close<static) qty=1.00 sl=1073.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 1071.00 | 1078.65 | 1079.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 1071.00 | 1078.65 | 1079.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 1071.00 | 1078.65 | 1079.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 1071.00 | 1078.65 | 1079.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1071.00 | 1078.65 | 1079.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 1059.70 | 1074.86 | 1077.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 1063.00 | 1062.10 | 1068.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 1065.70 | 1062.10 | 1068.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1075.20 | 1064.72 | 1069.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1075.20 | 1064.72 | 1069.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1070.00 | 1065.78 | 1069.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:00:00 | 1067.10 | 1066.94 | 1069.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 1108.40 | 1073.10 | 1071.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1108.40 | 1073.10 | 1071.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 10:15:00 | 1139.50 | 1086.38 | 1077.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 1125.80 | 1127.24 | 1111.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 14:00:00 | 1125.80 | 1127.24 | 1111.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1114.90 | 1123.03 | 1115.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 1114.90 | 1123.03 | 1115.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1111.30 | 1120.68 | 1115.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 1111.30 | 1120.68 | 1115.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1110.40 | 1118.62 | 1115.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1111.20 | 1118.62 | 1115.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1121.00 | 1118.33 | 1115.46 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 1111.00 | 1119.30 | 1119.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 14:15:00 | 1110.60 | 1117.56 | 1118.70 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 1136.90 | 1119.96 | 1119.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1157.50 | 1136.84 | 1130.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1144.80 | 1152.98 | 1146.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 1144.80 | 1152.98 | 1146.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1144.80 | 1152.98 | 1146.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 1144.80 | 1152.98 | 1146.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1150.00 | 1152.39 | 1146.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1161.00 | 1152.39 | 1146.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:30:00 | 1151.00 | 1150.66 | 1147.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:30:00 | 1151.00 | 1150.89 | 1148.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 1140.80 | 1149.49 | 1148.44 | SL hit (close<static) qty=1.00 sl=1142.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 1140.80 | 1149.49 | 1148.44 | SL hit (close<static) qty=1.00 sl=1142.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 1140.80 | 1149.49 | 1148.44 | SL hit (close<static) qty=1.00 sl=1142.40 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1139.40 | 1147.47 | 1147.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 15:15:00 | 1135.00 | 1141.82 | 1144.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1074.10 | 1071.96 | 1081.56 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1062.50 | 1071.96 | 1081.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1073.80 | 1066.87 | 1072.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 1073.80 | 1066.87 | 1072.92 | SL hit (close>ema400) qty=1.00 sl=1072.92 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1073.80 | 1066.87 | 1072.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1072.50 | 1067.99 | 1072.89 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1088.90 | 1076.36 | 1074.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1091.60 | 1084.64 | 1080.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 1084.10 | 1084.54 | 1080.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 1084.10 | 1084.54 | 1080.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1081.80 | 1083.99 | 1080.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1081.80 | 1083.99 | 1080.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1081.60 | 1083.51 | 1080.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 1082.30 | 1083.51 | 1080.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1081.30 | 1083.07 | 1080.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 1082.60 | 1083.07 | 1080.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1078.40 | 1082.09 | 1080.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1078.40 | 1082.09 | 1080.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1081.90 | 1082.05 | 1080.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1077.30 | 1082.05 | 1080.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1075.90 | 1080.82 | 1080.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 1075.00 | 1080.82 | 1080.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1070.70 | 1078.80 | 1079.54 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 1083.00 | 1079.64 | 1079.43 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1075.70 | 1078.85 | 1079.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 1074.50 | 1077.98 | 1078.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 1084.50 | 1073.33 | 1074.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1084.50 | 1073.33 | 1074.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1084.50 | 1073.33 | 1074.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 1088.10 | 1073.33 | 1074.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1079.90 | 1074.65 | 1074.68 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 1079.60 | 1075.64 | 1075.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 1084.70 | 1078.21 | 1076.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 15:15:00 | 1077.30 | 1078.67 | 1076.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:15:00 | 1092.80 | 1078.67 | 1076.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1098.00 | 1082.53 | 1078.90 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 1074.30 | 1080.38 | 1080.99 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1087.40 | 1081.91 | 1081.50 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 1076.00 | 1080.97 | 1081.15 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 1085.80 | 1080.84 | 1080.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1094.40 | 1085.15 | 1082.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 1108.70 | 1109.49 | 1101.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 12:00:00 | 1108.70 | 1109.49 | 1101.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1102.60 | 1108.40 | 1102.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1102.60 | 1108.40 | 1102.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1101.00 | 1106.92 | 1102.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1119.00 | 1106.92 | 1102.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 1087.90 | 1106.55 | 1108.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 1087.90 | 1106.55 | 1108.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 15:15:00 | 1080.00 | 1095.23 | 1102.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 1095.00 | 1093.46 | 1098.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:00:00 | 1095.00 | 1093.46 | 1098.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1104.50 | 1095.67 | 1099.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 1104.50 | 1095.67 | 1099.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1110.00 | 1098.54 | 1100.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 1109.90 | 1098.54 | 1100.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 1116.30 | 1102.09 | 1101.61 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1101.50 | 1106.20 | 1106.36 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1108.70 | 1106.70 | 1106.57 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1096.80 | 1104.72 | 1105.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1093.70 | 1102.52 | 1104.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 13:15:00 | 1095.00 | 1093.16 | 1097.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 14:00:00 | 1095.00 | 1093.16 | 1097.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1097.70 | 1094.07 | 1097.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1097.70 | 1094.07 | 1097.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1099.00 | 1095.05 | 1097.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1086.20 | 1095.05 | 1097.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1081.20 | 1092.28 | 1095.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 1079.50 | 1088.37 | 1093.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1025.52 | 1066.40 | 1079.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1004.10 | 1001.30 | 1013.26 | SL hit (close>ema200) qty=0.50 sl=1001.30 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 980.50 | 968.32 | 968.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 15:15:00 | 985.00 | 979.75 | 976.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 986.80 | 988.73 | 983.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 986.80 | 988.73 | 983.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 980.30 | 986.40 | 983.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 980.30 | 986.40 | 983.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 981.00 | 985.32 | 983.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 982.45 | 984.56 | 982.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 970.60 | 980.77 | 981.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 970.60 | 980.77 | 981.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 968.55 | 973.51 | 975.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 974.80 | 972.29 | 974.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 13:15:00 | 974.80 | 972.29 | 974.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 974.80 | 972.29 | 974.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 974.80 | 972.29 | 974.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 974.95 | 972.83 | 974.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 975.00 | 972.83 | 974.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 980.80 | 974.42 | 975.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 966.45 | 974.42 | 975.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 918.13 | 932.07 | 947.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 931.85 | 930.49 | 942.86 | SL hit (close>ema200) qty=0.50 sl=930.49 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 974.65 | 938.92 | 934.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 981.80 | 947.49 | 938.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 975.30 | 976.57 | 961.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:45:00 | 973.30 | 976.57 | 961.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 961.50 | 973.83 | 967.07 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 950.00 | 963.10 | 963.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 946.60 | 957.83 | 960.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 942.00 | 929.43 | 935.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 942.00 | 929.43 | 935.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 942.00 | 929.43 | 935.26 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 951.25 | 938.92 | 938.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 953.95 | 941.93 | 939.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 962.00 | 964.63 | 957.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 962.00 | 964.63 | 957.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 954.40 | 960.60 | 957.75 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 949.65 | 955.78 | 956.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 946.05 | 952.76 | 954.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 943.45 | 933.54 | 939.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 943.45 | 933.54 | 939.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 943.45 | 933.54 | 939.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 943.45 | 933.54 | 939.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 925.95 | 932.02 | 938.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 945.90 | 932.02 | 938.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 937.50 | 932.60 | 937.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 937.50 | 932.60 | 937.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 939.80 | 934.04 | 937.86 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 952.30 | 941.04 | 940.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 965.25 | 948.47 | 945.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 959.35 | 964.06 | 956.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 959.35 | 964.06 | 956.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 955.00 | 962.24 | 956.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 955.00 | 962.24 | 956.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 955.25 | 960.85 | 956.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 955.00 | 960.85 | 956.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 954.00 | 959.48 | 956.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:45:00 | 954.35 | 959.48 | 956.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 952.00 | 957.98 | 955.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 952.00 | 957.98 | 955.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 953.05 | 956.09 | 955.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 941.00 | 956.09 | 955.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 942.40 | 953.36 | 954.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 935.60 | 949.80 | 952.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 15:15:00 | 935.20 | 933.49 | 939.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 15:15:00 | 935.20 | 933.49 | 939.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 935.20 | 933.49 | 939.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 928.45 | 933.49 | 939.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 941.85 | 935.16 | 939.26 | SL hit (close>static) qty=1.00 sl=940.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 930.05 | 933.57 | 937.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 930.20 | 932.73 | 937.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:15:00 | 929.05 | 932.44 | 936.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 933.20 | 932.59 | 936.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 933.90 | 932.59 | 936.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 933.40 | 932.75 | 935.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 912.65 | 932.75 | 935.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 883.55 | 912.28 | 921.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 883.69 | 912.28 | 921.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 882.60 | 912.28 | 921.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 884.90 | 884.77 | 891.64 | SL hit (close>ema200) qty=0.50 sl=884.77 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 884.90 | 884.77 | 891.64 | SL hit (close>ema200) qty=0.50 sl=884.77 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 884.90 | 884.77 | 891.64 | SL hit (close>ema200) qty=0.50 sl=884.77 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 867.02 | 878.35 | 886.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 821.38 | 835.72 | 851.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 819.65 | 806.95 | 805.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 829.20 | 813.88 | 808.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 814.85 | 817.85 | 811.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 814.85 | 817.85 | 811.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 809.45 | 816.17 | 811.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 810.00 | 816.17 | 811.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 827.85 | 818.51 | 813.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 811.00 | 818.51 | 813.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 844.00 | 832.99 | 823.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 848.70 | 836.57 | 825.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:45:00 | 846.05 | 839.59 | 828.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 848.35 | 842.19 | 831.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 09:15:00 | 933.57 | 912.51 | 899.56 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-09 09:15:00 | 930.65 | 912.51 | 899.56 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-09 09:15:00 | 933.19 | 912.51 | 899.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 908.95 | 914.17 | 914.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 907.20 | 912.77 | 913.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 899.00 | 897.26 | 903.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 13:45:00 | 900.15 | 897.26 | 903.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 904.60 | 898.80 | 902.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 904.20 | 898.80 | 902.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 905.45 | 900.13 | 902.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 904.45 | 900.13 | 902.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 894.05 | 898.91 | 902.16 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 908.00 | 904.13 | 903.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 921.40 | 907.58 | 905.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 904.30 | 914.04 | 910.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 904.30 | 914.04 | 910.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 904.30 | 914.04 | 910.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 904.30 | 914.04 | 910.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 899.05 | 911.04 | 909.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 899.05 | 911.04 | 909.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 898.35 | 908.51 | 908.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 15:15:00 | 897.00 | 902.52 | 905.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 877.80 | 871.94 | 880.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 877.80 | 871.94 | 880.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 877.80 | 871.94 | 880.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 881.45 | 871.94 | 880.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 874.40 | 872.79 | 878.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 875.45 | 872.79 | 878.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 880.00 | 874.94 | 878.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 880.00 | 874.94 | 878.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 876.00 | 875.15 | 878.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 865.70 | 872.81 | 876.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 15:00:00 | 865.45 | 870.23 | 874.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 866.50 | 869.75 | 873.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 909.00 | 878.27 | 875.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 909.00 | 878.27 | 875.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 909.00 | 878.27 | 875.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 909.00 | 878.27 | 875.66 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 875.15 | 882.06 | 882.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 868.75 | 879.40 | 881.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 866.95 | 859.80 | 866.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 866.95 | 859.80 | 866.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 866.95 | 859.80 | 866.19 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 887.00 | 870.26 | 868.72 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 847.65 | 867.45 | 869.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 839.15 | 852.02 | 855.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 853.30 | 852.00 | 855.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 853.30 | 852.00 | 855.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 853.30 | 852.00 | 855.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 853.65 | 852.00 | 855.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 884.00 | 858.40 | 857.75 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 842.95 | 855.78 | 857.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 841.60 | 851.33 | 854.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 828.00 | 823.41 | 833.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 828.00 | 823.41 | 833.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 819.45 | 825.46 | 831.06 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 843.35 | 834.32 | 833.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 844.40 | 836.34 | 834.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 837.65 | 839.38 | 836.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 837.65 | 839.38 | 836.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 837.65 | 839.38 | 836.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 849.40 | 839.82 | 837.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:30:00 | 847.80 | 841.23 | 838.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 833.30 | 837.39 | 837.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 833.30 | 837.39 | 837.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 833.30 | 837.39 | 837.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 829.30 | 835.77 | 836.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 802.35 | 797.65 | 809.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 802.75 | 797.65 | 809.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 834.00 | 805.42 | 809.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 834.00 | 805.42 | 809.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 830.00 | 810.33 | 811.06 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 828.25 | 813.92 | 812.62 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 793.35 | 812.92 | 813.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 782.90 | 806.92 | 810.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 795.00 | 768.83 | 780.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 795.00 | 768.83 | 780.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 795.00 | 768.83 | 780.05 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 812.35 | 786.77 | 786.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 819.10 | 802.02 | 795.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 11:15:00 | 893.00 | 894.80 | 880.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 12:00:00 | 893.00 | 894.80 | 880.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 919.80 | 916.60 | 911.33 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 902.50 | 910.42 | 910.44 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 928.80 | 914.10 | 912.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 936.15 | 919.16 | 915.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 931.20 | 932.49 | 926.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 928.50 | 931.70 | 926.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 928.50 | 931.70 | 926.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 925.15 | 931.70 | 926.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 927.00 | 930.76 | 926.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 925.50 | 930.76 | 926.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 927.00 | 930.01 | 926.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:30:00 | 923.55 | 930.01 | 926.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 922.60 | 928.52 | 926.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 922.60 | 928.52 | 926.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 915.95 | 926.01 | 925.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 915.95 | 926.01 | 925.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 918.30 | 924.47 | 924.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 912.25 | 920.67 | 922.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 913.70 | 912.35 | 916.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 913.70 | 912.35 | 916.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 913.70 | 912.35 | 916.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 916.50 | 912.35 | 916.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 915.40 | 912.96 | 916.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 916.20 | 912.96 | 916.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 915.80 | 913.53 | 916.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 915.80 | 913.53 | 916.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 916.85 | 914.19 | 916.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 916.85 | 914.19 | 916.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 916.80 | 914.71 | 916.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 916.80 | 914.71 | 916.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 916.70 | 915.11 | 916.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 916.70 | 915.11 | 916.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 916.55 | 915.40 | 916.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 919.20 | 915.40 | 916.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 910.95 | 914.51 | 916.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 907.10 | 913.81 | 915.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 905.90 | 907.94 | 911.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 905.80 | 890.43 | 889.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 905.80 | 890.43 | 889.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 905.80 | 890.43 | 889.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 934.75 | 906.81 | 900.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 13:30:00 | 1260.70 | 2025-05-26 14:15:00 | 1386.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 14:45:00 | 1260.60 | 2025-05-26 14:15:00 | 1386.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 14:30:00 | 1267.30 | 2025-05-28 09:15:00 | 1276.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-06-02 12:00:00 | 1256.10 | 2025-06-03 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-06-02 13:00:00 | 1257.50 | 2025-06-03 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-09 11:15:00 | 1258.60 | 2025-06-09 14:15:00 | 1263.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-06-18 12:00:00 | 1196.70 | 2025-06-24 09:15:00 | 1207.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-18 13:45:00 | 1195.30 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-06-18 14:15:00 | 1195.20 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-19 09:30:00 | 1195.00 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1168.70 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-07-01 09:30:00 | 1222.70 | 2025-07-01 14:15:00 | 1206.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-01 12:30:00 | 1222.20 | 2025-07-01 14:15:00 | 1206.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1140.70 | 2025-07-29 09:15:00 | 1083.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:00:00 | 1141.90 | 2025-07-29 09:15:00 | 1084.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1141.70 | 2025-07-29 09:15:00 | 1084.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:00:00 | 1139.60 | 2025-07-29 09:15:00 | 1082.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1140.70 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-07-22 14:00:00 | 1141.90 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1141.70 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-07-23 10:00:00 | 1139.60 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2025-07-24 10:45:00 | 1134.20 | 2025-07-30 09:15:00 | 1123.00 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-07-31 11:15:00 | 1130.00 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-07-31 12:00:00 | 1129.20 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-31 12:30:00 | 1131.00 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-31 14:00:00 | 1132.60 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-08-08 12:15:00 | 1083.20 | 2025-08-11 09:15:00 | 1029.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 12:15:00 | 1083.20 | 2025-08-12 09:15:00 | 974.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-28 10:15:00 | 1095.90 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-08-28 11:30:00 | 1086.00 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-02 09:30:00 | 1091.20 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-02 14:45:00 | 1082.60 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-03 13:00:00 | 1097.40 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-03 13:45:00 | 1099.70 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-09-03 14:15:00 | 1097.40 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-04 09:45:00 | 1097.20 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-09-08 13:00:00 | 1067.10 | 2025-09-09 09:15:00 | 1108.40 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-09-23 09:15:00 | 1161.00 | 2025-09-24 10:15:00 | 1140.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-09-23 12:30:00 | 1151.00 | 2025-09-24 10:15:00 | 1140.80 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-23 13:30:00 | 1151.00 | 2025-09-24 10:15:00 | 1140.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest1 | 2025-10-01 09:15:00 | 1062.50 | 2025-10-01 15:15:00 | 1073.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1119.00 | 2025-10-28 12:15:00 | 1087.90 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-11-06 10:30:00 | 1079.50 | 2025-11-07 09:15:00 | 1025.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:30:00 | 1079.50 | 2025-11-12 09:15:00 | 1004.10 | STOP_HIT | 0.50 | 6.98% |
| BUY | retest2 | 2025-12-01 13:30:00 | 982.45 | 2025-12-02 09:15:00 | 970.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-05 09:15:00 | 966.45 | 2025-12-09 09:15:00 | 918.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:15:00 | 966.45 | 2025-12-09 12:15:00 | 931.85 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-01-08 09:15:00 | 928.45 | 2026-01-08 09:15:00 | 941.85 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-01-08 11:45:00 | 930.05 | 2026-01-12 09:15:00 | 883.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 12:30:00 | 930.20 | 2026-01-12 09:15:00 | 883.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:15:00 | 929.05 | 2026-01-12 09:15:00 | 882.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 930.05 | 2026-01-14 13:15:00 | 884.90 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-08 12:30:00 | 930.20 | 2026-01-14 13:15:00 | 884.90 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2026-01-08 14:15:00 | 929.05 | 2026-01-14 13:15:00 | 884.90 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-09 09:15:00 | 912.65 | 2026-01-16 09:15:00 | 867.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 912.65 | 2026-01-20 09:15:00 | 821.38 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-30 10:30:00 | 848.70 | 2026-02-09 09:15:00 | 933.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 12:45:00 | 846.05 | 2026-02-09 09:15:00 | 930.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 13:30:00 | 848.35 | 2026-02-09 09:15:00 | 933.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-24 13:15:00 | 865.70 | 2026-02-26 09:15:00 | 909.00 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2026-02-24 15:00:00 | 865.45 | 2026-02-26 09:15:00 | 909.00 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2026-02-25 09:15:00 | 866.50 | 2026-02-26 09:15:00 | 909.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2026-03-19 10:30:00 | 849.40 | 2026-03-20 13:15:00 | 833.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-03-19 12:30:00 | 847.80 | 2026-03-20 13:15:00 | 833.30 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-28 11:15:00 | 907.10 | 2026-05-07 09:15:00 | 905.80 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2026-04-29 09:45:00 | 905.90 | 2026-05-07 09:15:00 | 905.80 | STOP_HIT | 1.00 | 0.01% |
