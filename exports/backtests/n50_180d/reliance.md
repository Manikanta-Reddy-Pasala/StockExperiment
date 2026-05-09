# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1436.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
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

### Cycle 1 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.20 | 1501.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1370.70 | 1457.67 | 1477.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1449.75 | 1471.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:00:00 | 1459.10 | 1449.75 | 1471.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 1462.00 | 1450.57 | 1468.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1462.60 | 1451.01 | 1468.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 1462.20 | 1452.02 | 1468.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1388.90 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.47 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.09 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1315.80 | 1427.64 | 1448.33 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1316.34 | 1427.64 | 1448.33 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1315.98 | 1427.64 | 1448.33 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1455.60 | 1377.53 | 1392.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-02-27 09:15:00 | 1388.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-02-27 09:15:00 | 1389.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-02-27 09:15:00 | 1389.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-03-04 09:15:00 | 1315.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-03-04 09:15:00 | 1316.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-03-04 09:15:00 | 1315.98 | TARGET_HIT | 0.50 | 10.00% |
