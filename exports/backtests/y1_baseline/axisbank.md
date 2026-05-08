# AXISBANK (AXISBANK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1270.00
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
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 6 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 0
- **Avg / median % per leg:** 0.85% / 2.54%
- **Sum % (uncompounded):** 4.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 2 | 3 | 0 | 0.85% | 4.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 3 | 60.0% | 2 | 3 | 0 | 0.85% | 4.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 3 | 60.0% | 2 | 3 | 0 | 0.85% | 4.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 1209.00 | 1121.62 | 1121.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 1214.20 | 1125.05 | 1123.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1237.30 | 1260.94 | 1231.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1237.30 | 1260.94 | 1231.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1237.30 | 1260.94 | 1231.27 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 1286.00 | 1251.07 | 1235.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 1285.10 | 1251.41 | 1235.61 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-14 09:15:00 | 1288.10 | 1262.45 | 1245.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:15:00 | 1300.00 | 1262.82 | 1245.56 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-21 12:15:00 | 1286.50 | 1271.48 | 1252.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 13:15:00 | 1283.60 | 1271.60 | 1252.79 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 1322.50 | 1273.37 | 1255.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 1327.10 | 1273.91 | 1255.60 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Target hit | 2026-02-03 09:15:00 | 1413.61 | 1296.33 | 1271.33 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-03 09:15:00 | 1411.96 | 1296.33 | 1271.33 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1321.40 | 1348.68 | 1319.74 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-06 11:15:00 | 1326.80 | 1348.46 | 1319.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 1326.90 | 1348.24 | 1319.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 1316.20 | 1347.71 | 1319.82 | SL hit (close<static) qty=1.00 sl=1318.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1215.00 | 1328.39 | 1313.46 | SL hit (close<static) qty=1.00 sl=1231.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1215.00 | 1328.39 | 1313.46 | SL hit (close<static) qty=1.00 sl=1231.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1221.40 | 1299.88 | 1300.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1210.80 | 1299.00 | 1299.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.64 | 1270.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.37 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 1357.90 | 1286.29 | 1286.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 15:15:00 | 1364.00 | 1287.07 | 1286.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1311.89 | 1300.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1296.00 | 1311.73 | 1300.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1296.00 | 1311.73 | 1300.59 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-29 12:15:00 | 1304.20 | 1310.25 | 1300.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-29 13:15:00 | 1296.60 | 1310.11 | 1300.25 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-05 10:15:00 | 1285.10 | 2026-02-03 09:15:00 | 1413.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-14 10:15:00 | 1300.00 | 2026-02-03 09:15:00 | 1411.96 | TARGET_HIT | 1.00 | 8.61% |
| BUY | retest2 | 2026-01-21 13:15:00 | 1283.60 | 2026-03-06 14:15:00 | 1316.20 | STOP_HIT | 1.00 | 2.54% |
| BUY | retest2 | 2026-01-27 10:15:00 | 1327.10 | 2026-03-13 09:15:00 | 1215.00 | STOP_HIT | 1.00 | -8.45% |
| BUY | retest2 | 2026-03-06 12:15:00 | 1326.90 | 2026-03-13 09:15:00 | 1215.00 | STOP_HIT | 1.00 | -8.43% |
