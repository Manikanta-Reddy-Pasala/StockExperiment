# Bajaj Finserv Ltd. (BAJAJFINSV)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1818.30
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.65% / 3.72%
- **Sum % (uncompounded):** 15.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 1 | 3 | 2 | 2.65% | 15.9% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 1 | 3 | 2 | 2.65% | 15.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 1 | 3 | 2 | 2.65% | 15.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 00:00:00 | 1541.00 | 1507.72 | 1511.15 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=26.59 |
| Stop hit — per-position SL triggered | 2023-09-25 00:00:00 | 1577.40 | 1511.70 | 1535.96 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 1634.65 | 1515.47 | 1550.98 | Stage2 pullback-breakout RSI=67 vol=4.2x ATR=34.70 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 1635.05 | 1527.05 | 1605.76 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 1620.65 | 1535.83 | 1587.50 | Stage2 pullback-breakout RSI=60 vol=2.3x ATR=36.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 1694.37 | 1546.86 | 1629.53 | T1 booked 50% @ 1694.37 |
| Target hit | 2023-12-20 00:00:00 | 1680.90 | 1564.95 | 1684.01 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 1643.85 | 1586.88 | 1592.79 | Stage2 pullback-breakout RSI=62 vol=3.6x ATR=36.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 00:00:00 | 1716.61 | 1593.51 | 1639.66 | T1 booked 50% @ 1716.61 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1656.85 | 1595.18 | 1646.38 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-08 00:00:00 | 1541.00 | 2023-09-25 00:00:00 | 1577.40 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest1 | 2023-10-06 00:00:00 | 1634.65 | 2023-10-20 00:00:00 | 1635.05 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest1 | 2023-11-16 00:00:00 | 1620.65 | 2023-12-04 00:00:00 | 1694.37 | PARTIAL | 0.50 | 4.55% |
| BUY | retest1 | 2023-11-16 00:00:00 | 1620.65 | 2023-12-20 00:00:00 | 1680.90 | TARGET_HIT | 0.50 | 3.72% |
| BUY | retest1 | 2024-03-28 00:00:00 | 1643.85 | 2024-04-10 00:00:00 | 1716.61 | PARTIAL | 0.50 | 4.43% |
| BUY | retest1 | 2024-03-28 00:00:00 | 1643.85 | 2024-04-15 00:00:00 | 1656.85 | STOP_HIT | 0.50 | 0.79% |
