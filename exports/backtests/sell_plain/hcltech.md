# HCLTECH (HCLTECH)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 1182.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 4
- **Avg / median % per leg:** 15.00% / 15.09%
- **Sum % (uncompounded):** 60.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 0 | 4 | 15.00% | 60.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 0 | 4 | 15.00% | 60.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 4 | 100.0% | 0 | 0 | 4 | 15.00% | 60.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1447.10 | 1632.19 | 1632.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1370.40 | 1564.17 | 1594.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.21 | 1458.10 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1437.90 | 1399.30 | 1455.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.30 | 1455.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1433.70 | 1406.21 | 1454.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 11:15:00 | 1428.30 | 1406.66 | 1454.62 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1427.70 | 1408.40 | 1454.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 1422.90 | 1408.69 | 1454.01 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-17 10:15:00 | 1432.90 | 1414.72 | 1452.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:15:00 | 1434.70 | 1415.11 | 1452.67 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-20 09:15:00 | 1431.30 | 1416.00 | 1452.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-20 11:15:00 | 1438.90 | 1416.40 | 1452.22 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-20 13:15:00 | 1431.70 | 1416.75 | 1452.04 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 1424.40 | 1416.95 | 1451.79 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1449.40 | 1417.72 | 1451.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-21 12:15:00 | 1436.60 | 1417.91 | 1451.58 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-21 13:15:00 | 1442.00 | 1418.15 | 1451.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-21 15:15:00 | 1439.00 | 1418.59 | 1451.42 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1305.20 | 1417.46 | 1450.69 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 10:15:00 | 1219.50 | 1397.47 | 1437.95 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 11:15:00 | 1214.05 | 1395.62 | 1436.82 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 11:15:00 | 1210.74 | 1395.62 | 1436.82 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 12:15:00 | 1209.47 | 1393.69 | 1435.64 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-10 11:15:00 | 1428.30 | 2026-04-24 10:15:00 | 1219.50 | PARTIAL | 0.50 | 14.62% |
| SELL | retest2 | 2026-04-13 11:15:00 | 1422.90 | 2026-04-24 11:15:00 | 1214.05 | PARTIAL | 0.50 | 14.68% |
| SELL | retest2 | 2026-04-17 12:15:00 | 1434.70 | 2026-04-24 11:15:00 | 1210.74 | PARTIAL | 0.50 | 15.61% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-24 12:15:00 | 1209.47 | PARTIAL | 0.50 | 15.09% |
