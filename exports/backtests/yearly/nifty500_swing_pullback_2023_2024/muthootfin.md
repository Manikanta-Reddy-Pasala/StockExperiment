# Muthoot Finance Ltd. (MUTHOOTFIN)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2025-09-03 05:30:00 (745 bars)
- **Last close:** 2783.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 2.15% / 0.66%
- **Sum % (uncompounded):** 15.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.15% | 15.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.15% | 15.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.15% | 15.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 05:30:00 | 1288.00 | 1139.90 | 1287.96 | Stage2 pullback-breakout RSI=50 vol=1.7x ATR=28.24 |
| Stop hit — per-position SL triggered | 2023-09-11 05:30:00 | 1296.55 | 1153.28 | 1283.67 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 05:30:00 | 1271.60 | 1176.36 | 1246.67 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=26.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 05:30:00 | 1324.32 | 1181.83 | 1263.91 | T1 booked 50% @ 1324.32 |
| Stop hit — per-position SL triggered | 2023-11-10 05:30:00 | 1271.60 | 1193.11 | 1292.49 | SL hit (bars_held=14) |

### Cycle 3 — BUY (started 2024-01-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 05:30:00 | 1517.80 | 1265.59 | 1460.32 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=36.17 |
| Stop hit — per-position SL triggered | 2024-01-10 05:30:00 | 1463.54 | 1274.23 | 1468.50 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-02-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 05:30:00 | 1425.15 | 1297.10 | 1407.84 | Stage2 pullback-breakout RSI=52 vol=1.8x ATR=38.51 |
| Stop hit — per-position SL triggered | 2024-02-08 05:30:00 | 1367.38 | 1297.72 | 1403.26 | SL hit (bars_held=1) |

### Cycle 5 — BUY (started 2024-03-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 05:30:00 | 1434.45 | 1310.89 | 1358.61 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=46.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 05:30:00 | 1526.99 | 1319.46 | 1410.94 | T1 booked 50% @ 1526.99 |
| Target hit | 2024-05-09 05:30:00 | 1598.40 | 1394.61 | 1643.44 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-28 05:30:00 | 1288.00 | 2023-09-11 05:30:00 | 1296.55 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest1 | 2023-10-20 05:30:00 | 1271.60 | 2023-10-30 05:30:00 | 1324.32 | PARTIAL | 0.50 | 4.15% |
| BUY | retest1 | 2023-10-20 05:30:00 | 1271.60 | 2023-11-10 05:30:00 | 1271.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-04 05:30:00 | 1517.80 | 2024-01-10 05:30:00 | 1463.54 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest1 | 2024-02-07 05:30:00 | 1425.15 | 2024-02-08 05:30:00 | 1367.38 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest1 | 2024-03-21 05:30:00 | 1434.45 | 2024-04-01 05:30:00 | 1526.99 | PARTIAL | 0.50 | 6.45% |
| BUY | retest1 | 2024-03-21 05:30:00 | 1434.45 | 2024-05-09 05:30:00 | 1598.40 | TARGET_HIT | 0.50 | 11.43% |
