# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2025-11-10 09:15:00 → 2026-05-08 15:15:00 (854 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 5 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -2.71% / -2.44%
- **Sum % (uncompounded):** -10.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.71% | -10.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.71% | -10.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.71% | -10.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 1285.20 | 1249.59 | 1249.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1296.80 | 1250.34 | 1249.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1283.30 | 1283.44 | 1270.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 1274.60 | 1283.60 | 1270.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.60 | 1270.59 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 1284.00 | 1282.84 | 1270.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1284.80 | 1282.86 | 1270.91 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 1292.50 | 1282.95 | 1271.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:15:00 | 1298.40 | 1283.10 | 1271.20 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1298.00 | 1283.10 | 1271.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1294.60 | 1283.21 | 1272.07 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.22 | 1272.68 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.22 | 1272.68 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.22 | 1272.68 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1291.50 | 1281.22 | 1272.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1291.90 | 1281.33 | 1272.22 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1260.40 | 1282.77 | 1273.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1260.40 | 1282.77 | 1273.55 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |

### Cycle 2 — SELL (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 13:15:00 | 1181.40 | 1265.32 | 1265.63 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 1359.00 | 1260.91 | 1260.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 12:15:00 | 1364.50 | 1261.94 | 1261.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1274.74 | 1268.16 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1283.00 | 1274.74 | 1268.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1283.20 | 1274.83 | 1268.36 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-17 14:15:00 | 1284.80 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-03-18 10:15:00 | 1298.40 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1294.60 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1291.90 | 2026-03-30 09:15:00 | 1260.40 | STOP_HIT | 1.00 | -2.44% |
