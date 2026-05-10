# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2024-07-09 09:15:00 → 2026-05-08 15:15:00 (3168 bars)
- **Last close:** 902.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 0
- **Target hits / Stop hits / Partials:** 6 / 0 / 6
- **Avg / median % per leg:** 7.50% / 10.00%
- **Sum % (uncompounded):** 90.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 12 | 100.0% | 6 | 0 | 6 | 7.50% | 90.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 12 | 100.0% | 6 | 0 | 6 | 7.50% | 90.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 12 | 100.0% | 6 | 0 | 6 | 7.50% | 90.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 15:15:00 | 1355.00 | 1311.92 | 1311.92 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 1295.00 | 1311.75 | 1311.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1290.00 | 1311.54 | 1311.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 14:15:00 | 1310.60 | 1309.64 | 1310.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 1310.60 | 1309.64 | 1310.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1310.60 | 1309.64 | 1310.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 1310.50 | 1309.64 | 1310.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1317.50 | 1309.71 | 1310.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:15:00 | 1320.10 | 1309.71 | 1310.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1308.10 | 1309.69 | 1310.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:30:00 | 1307.80 | 1309.84 | 1310.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 1305.30 | 1309.99 | 1310.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:00:00 | 1305.20 | 1308.27 | 1309.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:45:00 | 1302.90 | 1308.19 | 1309.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1301.90 | 1304.79 | 1307.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 1304.50 | 1304.79 | 1307.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1299.90 | 1302.84 | 1306.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1302.00 | 1302.84 | 1306.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1307.80 | 1302.27 | 1306.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 1307.80 | 1302.27 | 1306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1302.30 | 1302.27 | 1306.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:15:00 | 1309.20 | 1302.27 | 1306.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1307.10 | 1302.32 | 1306.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 1309.80 | 1302.32 | 1306.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1307.40 | 1302.37 | 1306.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 1309.40 | 1302.37 | 1306.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1295.50 | 1302.30 | 1306.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 1305.30 | 1302.30 | 1306.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1296.90 | 1298.68 | 1303.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 1304.10 | 1298.68 | 1303.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1314.00 | 1298.83 | 1303.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1314.00 | 1298.83 | 1303.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1313.50 | 1298.98 | 1303.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 1313.70 | 1298.98 | 1303.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1306.10 | 1299.36 | 1303.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1299.80 | 1299.92 | 1304.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 1298.60 | 1299.95 | 1303.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 1242.41 | 1292.06 | 1299.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 1240.03 | 1292.06 | 1299.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 1239.94 | 1292.06 | 1299.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 1237.76 | 1292.06 | 1299.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 1234.81 | 1292.06 | 1299.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 15:15:00 | 1233.67 | 1291.57 | 1298.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-07 10:15:00 | 1177.02 | 1253.73 | 1275.22 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 10:15:00 | 1174.77 | 1253.73 | 1275.22 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 10:15:00 | 1174.68 | 1253.73 | 1275.22 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 10:15:00 | 1172.61 | 1253.73 | 1275.22 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 11:15:00 | 1169.82 | 1252.95 | 1274.72 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 11:15:00 | 1168.74 | 1252.95 | 1274.72 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-24 13:30:00 | 1307.80 | 2025-07-24 14:15:00 | 1242.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-25 12:00:00 | 1305.30 | 2025-07-24 14:15:00 | 1240.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-27 13:00:00 | 1305.20 | 2025-07-24 14:15:00 | 1239.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-27 14:45:00 | 1302.90 | 2025-07-24 14:15:00 | 1237.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1299.80 | 2025-07-24 14:15:00 | 1234.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 12:00:00 | 1298.60 | 2025-07-24 15:15:00 | 1233.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-24 13:30:00 | 1307.80 | 2025-08-07 10:15:00 | 1177.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-25 12:00:00 | 1305.30 | 2025-08-07 10:15:00 | 1174.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-27 13:00:00 | 1305.20 | 2025-08-07 10:15:00 | 1174.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-27 14:45:00 | 1302.90 | 2025-08-07 10:15:00 | 1172.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1299.80 | 2025-08-07 11:15:00 | 1169.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-17 12:00:00 | 1298.60 | 2025-08-07 11:15:00 | 1168.74 | TARGET_HIT | 0.50 | 10.00% |
