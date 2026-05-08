# INFY (INFY)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 1179.20
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
| ALERT3 | 1 |
| PENDING | 6 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 4 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 3.78% / 5.00%
- **Sum % (uncompounded):** 26.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.62% | 6.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.62% | 6.5% |
| SELL (all) | 3 | 3 | 100.0% | 1 | 0 | 2 | 6.67% | 20.0% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 0 | 2 | 6.67% | 20.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 0 | 2 | 6.67% | 20.0% |
| retest2 (combined) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.62% | 6.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 1528.80 | 1488.59 | 1488.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 1531.50 | 1489.41 | 1488.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 1493.60 | 1497.39 | 1493.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 12:15:00 | 1493.60 | 1497.39 | 1493.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1493.60 | 1497.39 | 1493.09 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-14 14:15:00 | 1503.60 | 1497.38 | 1493.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 1505.80 | 1497.46 | 1493.19 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-17 13:15:00 | 1506.70 | 1497.62 | 1493.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 14:15:00 | 1508.30 | 1497.73 | 1493.45 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 1490.00 | 1497.75 | 1493.50 | SL hit (close<static) qty=1.00 sl=1492.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 1490.00 | 1497.75 | 1493.50 | SL hit (close<static) qty=1.00 sl=1492.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-19 09:15:00 | 1528.00 | 1497.53 | 1493.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1537.70 | 1497.93 | 1493.75 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2025-12-22 09:15:00 | 1691.47 | 1574.07 | 1545.48 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-06 14:15:00 | 1506.50 | 1615.25 | 1599.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:15:00 | 1507.10 | 1614.17 | 1599.32 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1488.00 | 1598.23 | 1592.13 | SL hit (close<static) qty=1.00 sl=1492.20 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1399.50 | 1585.73 | 1586.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 1387.90 | 1581.91 | 1584.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.00 | 1311.02 | 1382.42 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1289.30 | 1314.96 | 1375.17 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1287.40 | 1314.68 | 1374.73 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1271.60 | 1311.05 | 1360.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1264.30 | 1310.59 | 1360.26 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1223.03 | 1303.14 | 1353.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1201.09 | 1303.14 | 1353.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-24 13:15:00 | 1158.66 | 1297.82 | 1349.63 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-14 15:15:00 | 1505.80 | 2025-11-18 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-11-17 14:15:00 | 1508.30 | 2025-11-18 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-11-19 10:15:00 | 1537.70 | 2025-12-22 09:15:00 | 1691.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-06 15:15:00 | 1507.10 | 2026-02-11 09:15:00 | 1488.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest1 | 2026-04-10 10:15:00 | 1287.40 | 2026-04-24 09:15:00 | 1223.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 10:15:00 | 1264.30 | 2026-04-24 09:15:00 | 1201.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-10 10:15:00 | 1287.40 | 2026-04-24 13:15:00 | 1158.66 | TARGET_HIT | 0.50 | 10.00% |
