# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 1198.40
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
| ALERT3 | 2 |
| PENDING | 6 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 9 / 0
- **Target hits / Stop hits / Partials:** 4 / 0 / 5
- **Avg / median % per leg:** 7.22% / 5.00%
- **Sum % (uncompounded):** 65.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 1453.40 | 1618.36 | 1619.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 1445.60 | 1584.19 | 1600.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.05 | 1455.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1437.90 | 1399.28 | 1453.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.28 | 1453.05 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1433.70 | 1406.13 | 1452.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1429.30 | 1406.36 | 1452.84 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1427.70 | 1408.33 | 1452.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1423.40 | 1408.48 | 1452.32 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-16 11:15:00 | 1435.40 | 1412.86 | 1451.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 12:15:00 | 1443.00 | 1413.16 | 1451.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-17 10:15:00 | 1432.90 | 1414.60 | 1451.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 1433.20 | 1414.79 | 1451.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-20 09:15:00 | 1430.00 | 1415.86 | 1450.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:15:00 | 1429.60 | 1416.00 | 1450.61 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 300m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1449.40 | 1416.91 | 1450.39 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1305.50 | 1416.72 | 1449.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1357.84 | 1416.72 | 1449.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1352.23 | 1416.72 | 1449.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1361.54 | 1416.72 | 1449.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1358.12 | 1416.72 | 1449.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1299.80 | 1415.55 | 1448.72 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-22 11:15:00 | 1289.88 | 1414.29 | 1447.92 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1286.37 | 1413.01 | 1447.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1286.64 | 1413.01 | 1447.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-23 09:15:00 | 1281.06 | 1407.93 | 1443.88 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1234.81 | 1398.59 | 1437.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-22 09:15:00 | 1357.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-22 09:15:00 | 1352.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-22 09:15:00 | 1361.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 14:15:00 | 1429.60 | 2026-04-22 09:15:00 | 1358.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-22 11:15:00 | 1289.88 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-22 12:15:00 | 1286.37 | TARGET_HIT | 0.50 | 9.63% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-22 12:15:00 | 1286.64 | TARGET_HIT | 0.50 | 10.23% |
| SELL | retest2 | 2026-04-20 14:15:00 | 1429.60 | 2026-04-23 09:15:00 | 1281.06 | TARGET_HIT | 0.50 | 10.39% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1299.80 | 2026-04-24 09:15:00 | 1234.81 | PARTIAL | 0.50 | 5.00% |
