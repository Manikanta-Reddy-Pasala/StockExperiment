# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2025-11-10 09:15:00 → 2026-05-08 15:15:00 (854 bars)
- **Last close:** 1198.00
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

### Cycle 1 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 1470.00 | 1636.36 | 1636.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1405.80 | 1634.06 | 1635.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.21 | 1458.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1437.90 | 1399.30 | 1455.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.30 | 1455.69 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1433.70 | 1406.21 | 1455.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1429.30 | 1406.44 | 1455.32 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1427.70 | 1408.40 | 1454.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1423.40 | 1408.55 | 1454.72 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-17 10:15:00 | 1432.90 | 1414.73 | 1453.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 1433.20 | 1414.91 | 1453.25 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-20 09:15:00 | 1431.30 | 1416.01 | 1452.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 10:15:00 | 1432.70 | 1416.17 | 1452.76 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1449.40 | 1417.72 | 1452.11 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-21 12:15:00 | 1436.60 | 1417.91 | 1452.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-21 13:15:00 | 1442.00 | 1418.15 | 1451.99 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-21 15:15:00 | 1439.00 | 1418.59 | 1451.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1305.20 | 1417.46 | 1451.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1357.83 | 1417.46 | 1451.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1352.23 | 1417.46 | 1451.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1361.54 | 1417.46 | 1451.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1361.07 | 1417.46 | 1451.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1289.88 | 1415.02 | 1449.57 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1289.43 | 1415.02 | 1449.57 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1286.37 | 1413.73 | 1448.76 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1281.06 | 1413.73 | 1448.76 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1239.94 | 1399.24 | 1439.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-22 09:15:00 | 1357.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-22 09:15:00 | 1352.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-22 09:15:00 | 1361.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 10:15:00 | 1432.70 | 2026-04-22 09:15:00 | 1361.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-22 11:15:00 | 1289.88 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-22 11:15:00 | 1289.43 | TARGET_HIT | 0.50 | 9.41% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-22 12:15:00 | 1286.37 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2026-04-20 10:15:00 | 1432.70 | 2026-04-22 12:15:00 | 1281.06 | TARGET_HIT | 0.50 | 10.58% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1305.20 | 2026-04-24 09:15:00 | 1239.94 | PARTIAL | 0.50 | 5.00% |
