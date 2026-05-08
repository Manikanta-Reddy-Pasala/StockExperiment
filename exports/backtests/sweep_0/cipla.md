# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1348.00
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
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 3 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 3 / 0 / 3
- **Avg / median % per leg:** 7.50% / 10.00%
- **Sum % (uncompounded):** 45.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1513.00 | 1531.12 | 1531.13 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1531.32 | 1531.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1568.00 | 1531.96 | 1531.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.41 | 1549.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 1546.50 | 1561.41 | 1549.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1546.50 | 1561.41 | 1549.20 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1524.90 | 1539.68 | 1539.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1518.00 | 1539.11 | 1539.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1530.00 | 1529.74 | 1533.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1532.20 | 1529.77 | 1533.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1532.20 | 1529.77 | 1533.75 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-11-28 15:15:00 | 1529.00 | 1529.76 | 1533.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1524.60 | 1529.71 | 1533.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-12-02 12:15:00 | 1517.00 | 1529.19 | 1533.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 1519.90 | 1529.10 | 1533.15 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 1503.10 | 1511.09 | 1517.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 1480.10 | 1510.78 | 1517.71 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1448.37 | 1502.17 | 1512.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 09:15:00 | 1443.90 | 1499.25 | 1510.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-16 09:15:00 | 1372.14 | 1490.86 | 1505.60 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 09:15:00 | 1367.91 | 1490.86 | 1505.60 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1406.09 | 1490.86 | 1505.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 1332.09 | 1454.95 | 1483.38 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-01 09:15:00 | 1524.60 | 2026-01-12 09:15:00 | 1448.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 13:15:00 | 1519.90 | 2026-01-13 09:15:00 | 1443.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:15:00 | 1524.60 | 2026-01-16 09:15:00 | 1372.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 13:15:00 | 1519.90 | 2026-01-16 09:15:00 | 1367.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 1480.10 | 2026-01-16 09:15:00 | 1406.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 1480.10 | 2026-01-23 13:15:00 | 1332.09 | TARGET_HIT | 0.50 | 10.00% |
