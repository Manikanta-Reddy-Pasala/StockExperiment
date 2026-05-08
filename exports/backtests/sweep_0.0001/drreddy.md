# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 1 |
| PENDING | 4 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -3.11% / -2.13%
- **Sum % (uncompounded):** -6.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.11% | -6.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.11% | -6.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.11% | -6.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 1210.40 | 1257.13 | 1257.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.40 | 1255.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.27 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-28 10:15:00 | 1224.00 | 1227.53 | 1239.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-28 11:15:00 | 1226.40 | 1227.52 | 1239.10 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 12:15:00 | 1223.70 | 1227.48 | 1239.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 1222.00 | 1227.43 | 1238.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-01 12:15:00 | 1221.60 | 1225.10 | 1236.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 1199.00 | 1224.84 | 1236.42 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1248.00 | 1224.13 | 1234.46 | SL hit (close>static) qty=1.00 sl=1247.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1248.00 | 1224.13 | 1234.46 | SL hit (close>static) qty=1.00 sl=1247.20 alert=retest2 |
| CROSSOVER_SKIP | 2026-02-17 14:15:00 | 1286.00 | 1241.91 | 1241.84 | min_gap filter: gap=0.005% < 0.010% |
| TREND_RESET | 2026-02-17 14:15:00 | 1286.00 | 1241.91 | 1241.84 | EMA inversion without crossover edge (EMA200=1241.91 EMA400=1241.84) — end cycle |
| CROSSOVER_SKIP | 2026-04-08 09:15:00 | 1187.90 | 1263.24 | 1263.27 | min_gap filter: gap=0.003% < 0.010% |

### Cycle 2 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.00 | 1259.94 | 1259.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1274.75 | 1267.46 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1283.00 | 1274.75 | 1267.60 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1283.20 | 1274.84 | 1267.68 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-28 13:15:00 | 1222.00 | 2026-02-05 12:15:00 | 1248.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-02-01 13:15:00 | 1199.00 | 2026-02-05 12:15:00 | 1248.00 | STOP_HIT | 1.00 | -4.09% |
