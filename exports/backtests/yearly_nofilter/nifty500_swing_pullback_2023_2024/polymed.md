# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1652.10
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 3
- **Avg / median % per leg:** 2.49% / 1.03%
- **Sum % (uncompounded):** 17.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 0 | 4 | 3 | 2.49% | 17.4% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 4 | 3 | 2.49% | 17.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 4 | 3 | 2.49% | 17.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 00:00:00 | 1424.75 | 1099.47 | 1385.17 | Stage2 pullback-breakout RSI=57 vol=3.6x ATR=58.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 00:00:00 | 1541.35 | 1106.99 | 1402.41 | T1 booked 50% @ 1541.35 |
| Stop hit — per-position SL triggered | 2023-09-20 00:00:00 | 1424.75 | 1113.61 | 1409.42 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 00:00:00 | 1405.80 | 1177.06 | 1378.60 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=49.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 00:00:00 | 1504.04 | 1194.09 | 1404.74 | T1 booked 50% @ 1504.04 |
| Stop hit — per-position SL triggered | 2023-11-13 00:00:00 | 1420.35 | 1201.25 | 1412.74 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 1614.60 | 1241.62 | 1505.22 | Stage2 pullback-breakout RSI=68 vol=1.5x ATR=55.90 |
| Stop hit — per-position SL triggered | 2023-12-15 00:00:00 | 1530.75 | 1272.88 | 1560.56 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-02-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 00:00:00 | 1480.20 | 1324.50 | 1453.06 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=47.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 00:00:00 | 1575.17 | 1330.95 | 1476.82 | T1 booked 50% @ 1575.17 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 1480.20 | 1341.85 | 1506.22 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 1633.20 | 1424.73 | 1563.45 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=53.29 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-13 00:00:00 | 1424.75 | 2023-09-15 00:00:00 | 1541.35 | PARTIAL | 0.50 | 8.18% |
| BUY | retest1 | 2023-09-13 00:00:00 | 1424.75 | 2023-09-20 00:00:00 | 1424.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-31 00:00:00 | 1405.80 | 2023-11-09 00:00:00 | 1504.04 | PARTIAL | 0.50 | 6.99% |
| BUY | retest1 | 2023-10-31 00:00:00 | 1405.80 | 2023-11-13 00:00:00 | 1420.35 | STOP_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2023-12-04 00:00:00 | 1614.60 | 2023-12-15 00:00:00 | 1530.75 | STOP_HIT | 1.00 | -5.19% |
| BUY | retest1 | 2024-02-01 00:00:00 | 1480.20 | 2024-02-06 00:00:00 | 1575.17 | PARTIAL | 0.50 | 6.42% |
| BUY | retest1 | 2024-02-01 00:00:00 | 1480.20 | 2024-02-13 00:00:00 | 1480.20 | STOP_HIT | 0.50 | 0.00% |
