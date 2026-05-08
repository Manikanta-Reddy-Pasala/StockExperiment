# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 1264.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 8 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -1.65% / -1.21%
- **Sum % (uncompounded):** -8.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.12% | -3.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.12% | -3.4% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.44% | -4.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.70% | -2.7% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.19% | -2.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.70% | -2.7% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.39% | -5.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 12:15:00 | 1420.10 | 1376.93 | 1376.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 1434.20 | 1379.01 | 1377.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.30 | 1386.09 | 1381.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.30 | 1386.09 | 1381.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.30 | 1386.09 | 1381.75 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 1344.20 | 1378.19 | 1378.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 1339.40 | 1377.46 | 1377.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.33 | 1376.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 1378.00 | 1375.33 | 1376.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1378.00 | 1375.33 | 1376.70 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-30 10:15:00 | 1364.90 | 1375.40 | 1376.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1360.30 | 1375.25 | 1376.63 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1390.10 | 1372.87 | 1375.32 | SL hit (close>static) qty=1.00 sl=1378.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 13:15:00 | 1399.90 | 1377.68 | 1377.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 15:15:00 | 1406.10 | 1379.66 | 1378.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.04 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 1398.20 | 1392.01 | 1386.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1398.70 | 1392.07 | 1386.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 13:15:00 | 1397.50 | 1392.20 | 1386.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-20 14:15:00 | 1394.20 | 1392.22 | 1386.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 1406.70 | 1392.38 | 1386.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 1401.70 | 1392.48 | 1386.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1384.80 | 1392.71 | 1387.03 | SL hit (close<static) qty=1.00 sl=1385.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1384.80 | 1392.71 | 1387.03 | SL hit (close<static) qty=1.00 sl=1385.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-25 09:15:00 | 1399.30 | 1392.70 | 1387.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-25 10:15:00 | 1394.90 | 1392.72 | 1387.12 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-25 11:15:00 | 1399.60 | 1392.79 | 1387.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 1400.00 | 1392.86 | 1387.24 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 1383.80 | 1393.60 | 1388.01 | SL hit (close<static) qty=1.00 sl=1385.70 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 1315.10 | 1382.74 | 1383.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.64 | 1382.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.44 | 1317.92 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 1290.30 | 1283.33 | 1317.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 12:15:00 | 1286.30 | 1283.36 | 1317.13 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1321.00 | 1283.72 | 1316.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1321.00 | 1283.72 | 1316.64 | SL hit (close>ema400) qty=1.00 sl=1316.64 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-29 13:15:00 | 1284.10 | 1313.14 | 1323.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 1280.10 | 1312.81 | 1323.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-30 11:15:00 | 1360.30 | 2026-02-03 09:15:00 | 1390.10 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1398.70 | 2026-02-24 14:15:00 | 1384.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-23 10:15:00 | 1401.70 | 2026-02-24 14:15:00 | 1384.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-02-25 12:15:00 | 1400.00 | 2026-02-27 12:15:00 | 1383.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest1 | 2026-04-09 12:15:00 | 1286.30 | 2026-04-10 09:15:00 | 1321.00 | STOP_HIT | 1.00 | -2.70% |
