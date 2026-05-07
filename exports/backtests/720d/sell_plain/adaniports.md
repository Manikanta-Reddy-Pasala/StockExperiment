# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1734.80
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 3 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 13.68% / 15.00%
- **Sum % (uncompounded):** 95.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 6 | 85.7% | 2 | 3 | 2 | 13.68% | 95.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 6 | 85.7% | 2 | 3 | 2 | 13.68% | 95.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 6 | 85.7% | 2 | 3 | 2 | 13.68% | 95.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 1430.50 | 1467.64 | 1467.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 1415.80 | 1466.28 | 1467.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 1473.00 | 1459.43 | 1463.07 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 1473.00 | 1459.43 | 1463.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1473.00 | 1459.43 | 1463.07 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-30 09:15:00 | 1451.40 | 1459.77 | 1463.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-09-30 10:15:00 | 1463.75 | 1459.81 | 1463.10 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-30 12:15:00 | 1451.90 | 1459.74 | 1463.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 1448.65 | 1459.49 | 1462.87 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-10-01 10:15:00 | 1453.25 | 1459.28 | 1462.72 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-01 11:15:00 | 1462.45 | 1459.32 | 1462.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-03 09:15:00 | 1451.95 | 1459.52 | 1462.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:15:00 | 1432.75 | 1459.19 | 1462.54 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 1231.35 | 1353.87 | 1388.57 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 1217.84 | 1353.87 | 1388.57 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2024-11-21 10:15:00 | 1014.05 | 1350.66 | 1386.78 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-11-21 10:15:00 | 1002.92 | 1350.66 | 1386.78 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1085.85 | 1154.57 | 1154.64 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 1301.40 | 1376.66 | 1376.84 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1403.00 | 1465.11 | 1465.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1396.30 | 1463.83 | 1464.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-02 10:15:00 | 1474.60 | 1510.11 | 1490.25 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 12:15:00 | 1438.50 | 1508.87 | 1489.83 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-03-06 15:15:00 | 1475.60 | 1499.39 | 1486.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1429.60 | 1498.70 | 1486.62 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 1362.40 | 1476.01 | 1476.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 1362.40 | 1476.01 | 1476.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.01 | 1476.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.47 | 1461.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 14:15:00 | 1452.10 | 1405.56 | 1430.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 1442.00 | 1406.38 | 1431.00 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1502.90 | 1417.20 | 1434.21 | SL hit (close>static) qty=1.00 sl=1487.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-30 14:15:00 | 1448.65 | 2024-11-21 09:15:00 | 1231.35 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-03 11:15:00 | 1432.75 | 2024-11-21 09:15:00 | 1217.84 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1448.65 | 2024-11-21 10:15:00 | 1014.05 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2024-10-03 11:15:00 | 1432.75 | 2024-11-21 10:15:00 | 1002.92 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2026-03-02 12:15:00 | 1438.50 | 2026-03-13 11:15:00 | 1362.40 | STOP_HIT | 1.00 | 5.29% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1429.60 | 2026-03-13 11:15:00 | 1362.40 | STOP_HIT | 1.00 | 4.70% |
| SELL | retest2 | 2026-04-09 09:15:00 | 1442.00 | 2026-04-15 09:15:00 | 1502.90 | STOP_HIT | 1.00 | -4.22% |
