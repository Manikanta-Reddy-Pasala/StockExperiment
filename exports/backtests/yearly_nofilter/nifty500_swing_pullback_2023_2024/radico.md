# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 3477.20
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 2 / 4 / 4
- **Avg / median % per leg:** 4.41% / 5.37%
- **Sum % (uncompounded):** 44.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 2 | 4 | 4 | 4.41% | 44.1% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 2 | 4 | 4 | 4.41% | 44.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 8 | 80.0% | 2 | 4 | 4 | 4.41% | 44.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 1278.10 | 1120.81 | 1205.87 | Stage2 pullback-breakout RSI=70 vol=2.9x ATR=34.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 1346.77 | 1122.89 | 1217.66 | T1 booked 50% @ 1346.77 |
| Target hit | 2023-08-07 00:00:00 | 1339.40 | 1178.86 | 1378.19 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 00:00:00 | 1290.35 | 1198.08 | 1280.54 | Stage2 pullback-breakout RSI=50 vol=1.9x ATR=42.92 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 1225.96 | 1199.88 | 1267.69 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 1279.25 | 1205.59 | 1231.94 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=40.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 00:00:00 | 1359.39 | 1209.26 | 1257.68 | T1 booked 50% @ 1359.39 |
| Target hit | 2023-12-19 00:00:00 | 1541.65 | 1288.88 | 1560.27 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 1668.35 | 1367.30 | 1634.98 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=48.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 00:00:00 | 1765.47 | 1387.44 | 1653.19 | T1 booked 50% @ 1765.47 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 1699.60 | 1407.26 | 1686.14 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-03-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 00:00:00 | 1626.70 | 1451.33 | 1592.33 | Stage2 pullback-breakout RSI=54 vol=4.6x ATR=54.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 00:00:00 | 1736.56 | 1463.61 | 1626.12 | T1 booked 50% @ 1736.56 |
| Stop hit — per-position SL triggered | 2024-04-08 00:00:00 | 1667.20 | 1477.10 | 1656.18 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 00:00:00 | 1768.65 | 1484.19 | 1672.72 | Stage2 pullback-breakout RSI=66 vol=3.6x ATR=56.65 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1683.67 | 1486.58 | 1677.64 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 1278.10 | 2023-07-04 00:00:00 | 1346.77 | PARTIAL | 0.50 | 5.37% |
| BUY | retest1 | 2023-07-03 00:00:00 | 1278.10 | 2023-08-07 00:00:00 | 1339.40 | TARGET_HIT | 0.50 | 4.80% |
| BUY | retest1 | 2023-09-06 00:00:00 | 1290.35 | 2023-09-12 00:00:00 | 1225.96 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest1 | 2023-11-02 00:00:00 | 1279.25 | 2023-11-07 00:00:00 | 1359.39 | PARTIAL | 0.50 | 6.26% |
| BUY | retest1 | 2023-11-02 00:00:00 | 1279.25 | 2023-12-19 00:00:00 | 1541.65 | TARGET_HIT | 0.50 | 20.51% |
| BUY | retest1 | 2024-01-24 00:00:00 | 1668.35 | 2024-02-05 00:00:00 | 1765.47 | PARTIAL | 0.50 | 5.82% |
| BUY | retest1 | 2024-01-24 00:00:00 | 1668.35 | 2024-02-13 00:00:00 | 1699.60 | STOP_HIT | 0.50 | 1.87% |
| BUY | retest1 | 2024-03-19 00:00:00 | 1626.70 | 2024-03-28 00:00:00 | 1736.56 | PARTIAL | 0.50 | 6.75% |
| BUY | retest1 | 2024-03-19 00:00:00 | 1626.70 | 2024-04-08 00:00:00 | 1667.20 | STOP_HIT | 0.50 | 2.49% |
| BUY | retest1 | 2024-04-12 00:00:00 | 1768.65 | 2024-04-15 00:00:00 | 1683.67 | STOP_HIT | 1.00 | -4.80% |
