# Reliance Industries (RELIANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4998 bars)
- **Last close:** 1437.90
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 8 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.66% / -1.26%
- **Sum % (uncompounded):** -6.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.20% | -2.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.20% | -2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.48% | -4.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.29% | -2.3% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.07% | -2.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.25% | -4.5% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.07% | -2.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.06 | 1398.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1482.40 | 1401.41 | 1400.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.16 | 1511.64 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-31 09:15:00 | 1548.00 | 1541.23 | 1511.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1555.00 | 1541.36 | 1512.18 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.52 | 1520.75 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1520.75 | 1550.52 | 1520.75 | SL hit qty=1.00 sl=1520.75 alert=retest1 |

### Cycle 2 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.19 | 1501.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1363.30 | 1460.02 | 1478.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1454.50 | 1474.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-05 13:15:00 | 1441.00 | 1454.07 | 1473.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-05 14:15:00 | 1444.50 | 1453.97 | 1472.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 1441.90 | 1453.85 | 1472.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1437.90 | 1453.69 | 1472.65 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-02-06 12:15:00 | 1440.50 | 1453.40 | 1472.22 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 13:15:00 | 1442.00 | 1453.29 | 1472.07 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1464.40 | 1455.06 | 1470.87 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1470.87 | 1455.06 | 1470.87 | SL hit qty=1.00 sl=1470.87 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 1457.20 | 1455.15 | 1470.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1454.70 | 1455.14 | 1470.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 1473.00 | 1377.61 | 1392.85 | SL hit qty=1.00 sl=1473.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-05 10:15:00 | 1452.30 | 1378.35 | 1393.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1460.10 | 1379.16 | 1393.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 13:15:00 | 1460.40 | 1380.83 | 1394.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-05 14:15:00 | 1464.40 | 1381.66 | 1394.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1455.50 | 1383.19 | 1395.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 1473.00 | 1383.19 | 1395.17 | SL hit qty=1.00 sl=1473.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1447.30 | 1383.83 | 1395.43 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-31 10:15:00 | 1555.00 | 2026-01-06 09:15:00 | 1520.75 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest1 | 2026-02-06 09:15:00 | 1437.90 | 2026-02-12 09:15:00 | 1470.87 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-12 12:15:00 | 1454.70 | 2026-05-05 09:15:00 | 1473.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-05 11:15:00 | 1460.10 | 2026-05-06 09:15:00 | 1473.00 | STOP_HIT | 1.00 | -0.88% |
