# ICICI Lombard General Insurance Company Ltd. (ICICIGI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1817.00
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 2.03% / -0.66%
- **Sum % (uncompounded):** 10.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.03% | 10.1% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.03% | 10.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.03% | 10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 00:00:00 | 1398.20 | 1216.28 | 1361.17 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=34.53 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 1346.40 | 1222.84 | 1367.99 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 1377.85 | 1248.43 | 1351.57 | Stage2 pullback-breakout RSI=60 vol=1.5x ATR=27.49 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 1336.61 | 1252.93 | 1354.90 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 00:00:00 | 1371.70 | 1263.86 | 1333.00 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=28.08 |
| Stop hit — per-position SL triggered | 2023-11-03 00:00:00 | 1362.70 | 1273.92 | 1354.96 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-01-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-17 00:00:00 | 1454.45 | 1332.23 | 1407.77 | Stage2 pullback-breakout RSI=61 vol=7.2x ATR=32.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 00:00:00 | 1520.12 | 1343.79 | 1450.42 | T1 booked 50% @ 1520.12 |
| Target hit | 2024-03-13 00:00:00 | 1643.25 | 1421.56 | 1645.07 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-07 00:00:00 | 1398.20 | 2023-08-11 00:00:00 | 1346.40 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest1 | 2023-09-15 00:00:00 | 1377.85 | 2023-09-22 00:00:00 | 1336.61 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2023-10-19 00:00:00 | 1371.70 | 2023-11-03 00:00:00 | 1362.70 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2024-01-17 00:00:00 | 1454.45 | 2024-01-30 00:00:00 | 1520.12 | PARTIAL | 0.50 | 4.52% |
| BUY | retest1 | 2024-01-17 00:00:00 | 1454.45 | 2024-03-13 00:00:00 | 1643.25 | TARGET_HIT | 0.50 | 12.98% |
