# AXISBANK (AXISBANK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1296.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 7 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 6 / 0
- **Avg / median % per leg:** 0.39% / -1.84%
- **Sum % (uncompounded):** 2.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 6 | 0 | 0.39% | 2.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.84% | -1.8% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 5 | 0 | 0.83% | 4.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.84% | -1.8% |
| retest2 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | 0.83% | 4.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 15:15:00 | 1235.20 | 1192.40 | 1192.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 1254.30 | 1193.01 | 1192.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1201.15 | 1219.13 | 1207.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1201.15 | 1219.13 | 1207.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1201.15 | 1219.13 | 1207.90 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 1098.00 | 1044.57 | 1044.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 1106.90 | 1060.18 | 1053.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1177.10 | 1180.86 | 1149.00 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-06 10:15:00 | 1192.40 | 1179.09 | 1150.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 1196.60 | 1179.26 | 1151.20 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1174.60 | 1205.86 | 1180.78 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 1174.60 | 1205.86 | 1180.78 | SL hit (close<ema400) qty=1.00 sl=1180.78 alert=retest1 |

### Cycle 3 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1129.80 | 1110.78 | 1110.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 1133.60 | 1111.01 | 1110.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1237.30 | 1260.94 | 1230.37 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 1236.80 | 1260.43 | 1230.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1236.80 | 1260.43 | 1230.42 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-30 09:15:00 | 1240.40 | 1246.31 | 1230.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1240.50 | 1246.25 | 1230.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1215.00 | 1328.39 | 1313.35 | SL hit (close<static) qty=1.00 sl=1228.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 1242.50 | 1306.37 | 1303.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:15:00 | 1247.40 | 1305.78 | 1302.94 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 1212.20 | 1302.30 | 1301.26 | SL hit (close<static) qty=1.00 sl=1228.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 13:15:00 | 1238.70 | 1250.70 | 1271.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 1247.10 | 1250.66 | 1271.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 1223.60 | 1250.08 | 1270.82 | SL hit (close<static) qty=1.00 sl=1228.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-07 13:15:00 | 1240.60 | 1249.63 | 1270.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 1250.90 | 1249.64 | 1270.19 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1247.00 | 1249.61 | 1270.07 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 1326.40 | 1251.03 | 1270.58 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-17 14:15:00 | 1357.90 | 1286.29 | 1286.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 14:15:00 | 1357.90 | 1286.29 | 1286.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 1357.90 | 1286.29 | 1286.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 15:15:00 | 1364.00 | 1287.07 | 1286.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1311.89 | 1300.57 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1296.00 | 1311.73 | 1300.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1296.00 | 1311.73 | 1300.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-29 12:15:00 | 1304.20 | 1310.25 | 1300.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-29 13:15:00 | 1296.60 | 1310.11 | 1300.21 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-06 11:15:00 | 1196.60 | 2025-07-01 09:15:00 | 1174.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1240.50 | 2026-03-13 09:15:00 | 1215.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-03-18 10:15:00 | 1247.40 | 2026-03-19 09:15:00 | 1212.20 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1247.10 | 2026-04-07 10:15:00 | 1223.60 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-04-07 14:15:00 | 1250.90 | 2026-04-17 14:15:00 | 1357.90 | STOP_HIT | 1.00 | 8.55% |
| BUY | retest2 | 2026-04-08 10:15:00 | 1326.40 | 2026-04-17 14:15:00 | 1357.90 | STOP_HIT | 1.00 | 2.37% |
