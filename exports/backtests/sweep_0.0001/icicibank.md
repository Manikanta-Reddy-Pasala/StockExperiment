# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1267.80
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
| ALERT2_SKIP | 1 |
| ALERT3 | 1 |
| PENDING | 6 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.89% / 1.41%
- **Sum % (uncompounded):** 3.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.89% | 3.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.89% | 3.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.89% | 3.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 1402.00 | 1438.59 | 1438.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 1399.30 | 1438.20 | 1438.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1424.40 | 1419.56 | 1427.19 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-15 15:15:00 | 1418.00 | 1419.70 | 1427.04 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-16 09:15:00 | 1421.20 | 1419.71 | 1427.01 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-09-17 15:15:00 | 1417.20 | 1419.83 | 1426.61 | ENTRY1 cross detected — sustain check pending (15m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1424.40 | 1419.87 | 1426.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1424.40 | 1419.87 | 1426.60 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-18 13:15:00 | 1420.30 | 1419.97 | 1426.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-18 14:15:00 | 1422.20 | 1420.00 | 1426.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-19 09:15:00 | 1405.10 | 1419.88 | 1426.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1403.80 | 1419.72 | 1426.26 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 1436.40 | 1392.90 | 1404.36 | SL hit (close>static) qty=1.00 sl=1432.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-20 09:15:00 | 1407.20 | 1393.48 | 1404.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 1401.80 | 1393.56 | 1404.52 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 1331.71 | 1377.11 | 1392.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 1382.00 | 1368.34 | 1384.78 | SL hit (close>ema200) qty=0.50 sl=1368.34 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 1412.90 | 1373.87 | 1375.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1403.00 | 1374.16 | 1375.93 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2026-01-12 14:15:00 | 1410.60 | 1377.69 | 1377.65 | min_gap filter: gap=0.003% < 0.010% |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 1410.60 | 1377.69 | 1377.65 | Force close (TREND_INVERSION) qty=1.00 alert=retest2 |
| TREND_RESET | 2026-01-12 14:15:00 | 1410.60 | 1377.69 | 1377.65 | EMA inversion without crossover edge (EMA200=1377.69 EMA400=1377.65) — end cycle |
| CROSSOVER_SKIP | 2026-01-23 12:15:00 | 1354.50 | 1378.88 | 1378.91 | min_gap filter: gap=0.002% < 0.010% |
| CROSSOVER_SKIP | 2026-02-06 13:15:00 | 1403.70 | 1377.86 | 1377.78 | min_gap filter: gap=0.006% < 0.010% |
| CROSSOVER_SKIP | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.08 | min_gap filter: gap=0.005% < 0.010% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-19 10:15:00 | 1403.80 | 2025-10-17 14:15:00 | 1436.40 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-20 10:15:00 | 1401.80 | 2025-11-06 09:15:00 | 1331.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-20 10:15:00 | 1401.80 | 2025-11-13 09:15:00 | 1382.00 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2026-01-09 10:15:00 | 1403.00 | 2026-01-12 14:15:00 | 1410.60 | STOP_HIT | 1.00 | -0.54% |
