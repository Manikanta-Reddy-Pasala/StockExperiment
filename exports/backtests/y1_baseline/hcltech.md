# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1198.00
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
| ALERT3 | 3 |
| PENDING | 9 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 6 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 9 / 2
- **Target hits / Stop hits / Partials:** 4 / 2 / 5
- **Avg / median % per leg:** 5.33% / 5.00%
- **Sum % (uncompounded):** 58.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.16% | -6.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.59% | -2.6% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.72% | -3.7% |
| SELL (all) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.59% | -2.6% |
| retest2 (combined) | 10 | 9 | 90.0% | 4 | 1 | 5 | 6.13% | 61.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 14:15:00 | 1525.20 | 1500.85 | 1500.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 1532.00 | 1501.91 | 1501.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 13:15:00 | 1632.60 | 1639.90 | 1603.48 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-01 11:15:00 | 1643.50 | 1637.53 | 1605.53 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1642.30 | 1637.58 | 1605.72 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-02 14:15:00 | 1641.70 | 1637.55 | 1607.10 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-02 15:15:00 | 1640.00 | 1637.57 | 1607.27 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1599.70 | 1637.20 | 1607.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 1599.70 | 1637.20 | 1607.23 | SL hit (close<ema400) qty=1.00 sl=1607.23 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 1642.00 | 1634.18 | 1607.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 1640.30 | 1634.24 | 1607.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-10 13:15:00 | 1579.30 | 1655.03 | 1640.08 | SL hit (close<static) qty=1.00 sl=1580.10 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-13 14:15:00)

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
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1357.83 | 1417.45 | 1450.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1352.23 | 1417.45 | 1450.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1361.54 | 1417.45 | 1450.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1361.07 | 1417.45 | 1450.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1289.88 | 1415.00 | 1448.49 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1289.43 | 1415.00 | 1448.49 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1286.37 | 1413.72 | 1447.68 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1281.06 | 1413.72 | 1447.68 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1239.94 | 1399.23 | 1438.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-01 12:15:00 | 1642.30 | 2026-01-05 09:15:00 | 1599.70 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2026-01-07 10:15:00 | 1640.30 | 2026-02-10 13:15:00 | 1579.30 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-22 09:15:00 | 1357.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-22 09:15:00 | 1352.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-22 09:15:00 | 1361.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 10:15:00 | 1432.70 | 2026-04-22 09:15:00 | 1361.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-22 11:15:00 | 1289.88 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-22 11:15:00 | 1289.43 | TARGET_HIT | 0.50 | 9.41% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-22 12:15:00 | 1286.37 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2026-04-20 10:15:00 | 1432.70 | 2026-04-22 12:15:00 | 1281.06 | TARGET_HIT | 0.50 | 10.58% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1305.20 | 2026-04-24 09:15:00 | 1239.94 | PARTIAL | 0.50 | 5.00% |
