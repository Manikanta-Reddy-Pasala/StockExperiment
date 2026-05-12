# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1512.90
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** 0.04% / 0.27%
- **Sum % (uncompounded):** 0.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.04% | 0.2% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.04% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.04% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 00:00:00 | 1373.00 | 1079.07 | 1294.92 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=49.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 00:00:00 | 1472.36 | 1098.71 | 1352.15 | T1 booked 50% @ 1472.36 |
| Stop hit — per-position SL triggered | 2023-08-10 00:00:00 | 1398.90 | 1122.23 | 1398.29 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 1458.30 | 1163.49 | 1404.73 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=41.19 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 1396.51 | 1179.52 | 1418.95 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-01-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 00:00:00 | 1729.10 | 1397.82 | 1642.83 | Stage2 pullback-breakout RSI=63 vol=4.3x ATR=56.89 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 1643.76 | 1424.28 | 1675.14 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 1478.15 | 1447.14 | 1423.43 | Stage2 pullback-breakout RSI=57 vol=2.6x ATR=55.23 |
| Stop hit — per-position SL triggered | 2024-04-22 00:00:00 | 1482.15 | 1448.05 | 1443.83 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-24 00:00:00 | 1373.00 | 2023-08-01 00:00:00 | 1472.36 | PARTIAL | 0.50 | 7.24% |
| BUY | retest1 | 2023-07-24 00:00:00 | 1373.00 | 2023-08-10 00:00:00 | 1398.90 | STOP_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2023-09-04 00:00:00 | 1458.30 | 2023-09-12 00:00:00 | 1396.51 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest1 | 2024-01-05 00:00:00 | 1729.10 | 2024-01-18 00:00:00 | 1643.76 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest1 | 2024-04-04 00:00:00 | 1478.15 | 2024-04-22 00:00:00 | 1482.15 | STOP_HIT | 1.00 | 0.27% |
