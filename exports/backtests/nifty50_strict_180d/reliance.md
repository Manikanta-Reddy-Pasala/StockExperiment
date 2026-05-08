# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 1435.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 8 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 4 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 2 / 1 / 3
- **Avg / median % per leg:** 6.00% / 5.00%
- **Sum % (uncompounded):** 36.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 6 | 100.0% | 2 | 1 | 3 | 6.00% | 36.0% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 6.81% | 13.6% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.60% | 22.4% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 6.81% | 13.6% |
| retest2 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.60% | 22.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 09:15:00 | 1416.20 | 1495.64 | 1495.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 10:15:00 | 1400.10 | 1494.69 | 1495.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1454.49 | 1472.13 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-05 13:15:00 | 1441.00 | 1454.06 | 1470.97 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-05 14:15:00 | 1444.50 | 1453.96 | 1470.83 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 1441.90 | 1453.84 | 1470.69 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1437.90 | 1453.68 | 1470.53 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-02-06 12:15:00 | 1440.50 | 1453.39 | 1470.13 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 13:15:00 | 1442.00 | 1453.28 | 1469.99 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.10 | 1453.92 | 1469.50 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-10 11:15:00 | 1459.60 | 1454.09 | 1469.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:15:00 | 1460.00 | 1454.15 | 1469.38 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-12 10:15:00 | 1462.20 | 1455.12 | 1468.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 1457.20 | 1455.14 | 1468.93 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1366.01 | 1434.32 | 1452.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1387.00 | 1434.32 | 1452.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1384.34 | 1434.32 | 1452.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1314.00 | 1428.91 | 1448.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1311.48 | 1428.91 | 1448.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 1425.00 | 1419.82 | 1441.44 | SL hit (close>ema200) qty=0.50 sl=1419.82 alert=retest1 |
| Cross detected — sustain check pending | 2026-05-05 10:15:00 | 1452.30 | 1378.35 | 1392.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1460.10 | 1379.16 | 1393.18 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 13:15:00 | 1460.40 | 1380.83 | 1393.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-05 14:15:00 | 1464.40 | 1381.66 | 1394.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-05 15:15:00 | 1463.00 | 1382.47 | 1394.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1455.50 | 1383.19 | 1394.87 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-06 09:15:00 | 1437.90 | 2026-03-02 09:15:00 | 1366.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 12:15:00 | 1460.00 | 2026-03-02 09:15:00 | 1387.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1457.20 | 2026-03-02 09:15:00 | 1384.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-06 09:15:00 | 1437.90 | 2026-03-04 09:15:00 | 1314.00 | TARGET_HIT | 0.50 | 8.62% |
| SELL | retest2 | 2026-02-10 12:15:00 | 1460.00 | 2026-03-04 09:15:00 | 1311.48 | TARGET_HIT | 0.50 | 10.17% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1457.20 | 2026-03-09 14:15:00 | 1425.00 | STOP_HIT | 0.50 | 2.21% |
