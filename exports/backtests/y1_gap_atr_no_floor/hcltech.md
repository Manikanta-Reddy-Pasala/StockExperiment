# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1198.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 0%)
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
| ALERT3 | 2 |
| PENDING | 6 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 0
- **Target hits / Stop hits / Partials:** 5 / 0 / 0
- **Avg / median % per leg:** 3.14% / 2.97%
- **Sum % (uncompounded):** 15.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 5 | 100.0% | 5 | 0 | 0 | 3.14% | 15.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 5 | 0 | 0 | 3.14% | 15.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 5 | 100.0% | 5 | 0 | 0 | 3.14% | 15.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1454.70 | 1625.22 | 1626.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1450.70 | 1587.14 | 1605.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.18 | 1457.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1437.90 | 1399.27 | 1454.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.27 | 1454.19 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1433.70 | 1406.19 | 1454.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1429.30 | 1406.42 | 1453.93 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1427.70 | 1408.38 | 1453.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1423.40 | 1408.53 | 1453.37 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-17 10:15:00 | 1432.90 | 1414.71 | 1452.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 1433.20 | 1414.89 | 1452.04 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-20 09:15:00 | 1431.30 | 1415.99 | 1451.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 10:15:00 | 1432.70 | 1416.16 | 1451.59 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1449.40 | 1417.71 | 1450.99 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-21 12:15:00 | 1436.60 | 1417.90 | 1450.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-21 13:15:00 | 1442.00 | 1418.14 | 1450.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-21 15:15:00 | 1439.00 | 1418.57 | 1450.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1305.20 | 1417.45 | 1450.04 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Target hit | 2026-04-22 09:15:00 | 1384.92 | 1417.45 | 1450.04 | Target hit (0%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-22 09:15:00 | 1381.13 | 1417.45 | 1450.04 | Target hit (0%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-22 09:15:00 | 1395.25 | 1417.45 | 1450.04 | Target hit (0%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-22 09:15:00 | 1397.87 | 1417.45 | 1450.04 | Target hit (0%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-24 09:15:00 | 1245.66 | 1399.23 | 1438.43 | Target hit (0%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-22 09:15:00 | 1384.92 | TARGET_HIT | 1.00 | 3.11% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-22 09:15:00 | 1381.13 | TARGET_HIT | 1.00 | 2.97% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-22 09:15:00 | 1395.25 | TARGET_HIT | 1.00 | 2.65% |
| SELL | retest2 | 2026-04-20 10:15:00 | 1432.70 | 2026-04-22 09:15:00 | 1397.87 | TARGET_HIT | 1.00 | 2.43% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1305.20 | 2026-04-24 09:15:00 | 1245.66 | TARGET_HIT | 1.00 | 4.56% |
