# RELIANCE (RELIANCE)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1435.50
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
| PENDING | 5 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.03% / -3.03%
- **Sum % (uncompounded):** -9.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.03% | -9.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.03% | -9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.03% | -9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 1300.00 | 1249.14 | 1248.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 1302.30 | 1249.67 | 1249.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1475.70 | 1481.30 | 1439.08 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-16 11:15:00 | 1482.80 | 1481.30 | 1439.51 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-16 12:15:00 | 1480.70 | 1481.29 | 1439.71 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-16 13:15:00 | 1485.00 | 1481.33 | 1439.94 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 14:15:00 | 1485.40 | 1481.37 | 1440.16 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-17 10:15:00 | 1481.80 | 1481.41 | 1440.80 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:15:00 | 1482.80 | 1481.42 | 1441.01 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1446.90 | 1480.53 | 1442.91 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1437.90 | 1479.76 | 1442.90 | SL hit (close<ema400) qty=1.00 sl=1442.90 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1437.90 | 1479.76 | 1442.90 | SL hit (close<ema400) qty=1.00 sl=1442.90 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-23 09:15:00 | 1477.50 | 1392.99 | 1395.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-23 10:15:00 | 1463.10 | 1393.69 | 1396.32 | ENTRY2 sustain failed after 60m |

### Cycle 2 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.01 | 1398.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1483.00 | 1401.36 | 1400.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.17 | 1511.64 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-31 09:15:00 | 1548.00 | 1541.24 | 1511.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1555.00 | 1541.37 | 1512.18 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.51 | 1520.74 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 1510.50 | 1550.11 | 1520.69 | SL hit (close<ema400) qty=1.00 sl=1520.69 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-16 14:15:00 | 1485.40 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest1 | 2025-07-17 11:15:00 | 1482.80 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2025-12-31 10:15:00 | 1555.00 | 2026-01-06 10:15:00 | 1510.50 | STOP_HIT | 1.00 | -2.86% |
