# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 949.90
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 1
- **Avg / median % per leg:** -0.46% / -2.32%
- **Sum % (uncompounded):** -4.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.57% | -10.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.57% | -10.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.23% | 6.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.23% | 6.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.57% | -10.3% |
| retest2 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.23% | 6.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1277.00 | 1227.16 | 1227.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1294.00 | 1229.39 | 1228.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1246.00 | 1247.94 | 1239.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1263.00 | 1247.93 | 1240.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 12:30:00 | 1256.20 | 1248.25 | 1240.32 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 14:30:00 | 1262.10 | 1248.46 | 1240.50 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 13:00:00 | 1256.40 | 1249.08 | 1241.01 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1242.90 | 1249.58 | 1241.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1242.90 | 1249.58 | 1241.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1245.40 | 1249.54 | 1241.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1241.60 | 1249.54 | 1241.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.73 | SL hit (close<ema400) qty=1.00 sl=1241.73 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.73 | SL hit (close<ema400) qty=1.00 sl=1241.73 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.73 | SL hit (close<ema400) qty=1.00 sl=1241.73 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.73 | SL hit (close<ema400) qty=1.00 sl=1241.73 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1227.00 | 1249.31 | 1241.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1231.90 | 1249.14 | 1241.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 1229.00 | 1249.14 | 1241.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1179.00 | 1234.98 | 1235.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1172.00 | 1234.35 | 1234.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1224.10 | 1223.55 | 1229.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 1224.10 | 1223.55 | 1229.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1224.10 | 1223.55 | 1229.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 1231.00 | 1223.55 | 1229.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1225.70 | 1222.94 | 1228.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1227.20 | 1222.94 | 1228.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1229.70 | 1222.74 | 1228.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 1233.30 | 1222.74 | 1228.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1230.30 | 1222.82 | 1228.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 1232.50 | 1222.82 | 1228.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1215.10 | 1222.54 | 1227.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 1209.00 | 1222.38 | 1227.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 14:15:00 | 1148.55 | 1200.82 | 1213.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-29 09:15:00 | 1088.10 | 1171.18 | 1193.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 916.55 | 879.60 | 879.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 934.75 | 886.21 | 883.32 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:30:00 | 1202.00 | 2025-05-13 13:15:00 | 1229.00 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-05-12 14:00:00 | 1207.10 | 2025-05-13 13:15:00 | 1229.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-05-13 09:15:00 | 1200.20 | 2025-05-16 09:15:00 | 1257.80 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest1 | 2025-06-09 09:30:00 | 1263.00 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest1 | 2025-06-09 12:30:00 | 1256.20 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest1 | 2025-06-09 14:30:00 | 1262.10 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-06-10 13:00:00 | 1256.40 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-07-01 14:15:00 | 1209.00 | 2025-07-17 14:15:00 | 1148.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 14:15:00 | 1209.00 | 2025-07-29 09:15:00 | 1088.10 | TARGET_HIT | 0.50 | 10.00% |
