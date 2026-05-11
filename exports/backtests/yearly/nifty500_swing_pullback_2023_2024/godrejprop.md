# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1873.60
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
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 2.06% / 1.47%
- **Sum % (uncompounded):** 10.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | 2.06% | 10.3% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 2.06% | 10.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | 2.06% | 10.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 05:30:00 | 1598.70 | 1322.63 | 1523.88 | Stage2 pullback-breakout RSI=69 vol=1.8x ATR=42.59 |
| Stop hit — per-position SL triggered | 2023-07-20 05:30:00 | 1622.20 | 1350.05 | 1580.69 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 05:30:00 | 1675.00 | 1456.97 | 1599.28 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=45.41 |
| Stop hit — per-position SL triggered | 2023-10-20 05:30:00 | 1649.50 | 1478.96 | 1653.96 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 05:30:00 | 1716.00 | 1490.93 | 1647.53 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=46.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 05:30:00 | 1808.41 | 1497.13 | 1676.08 | T1 booked 50% @ 1808.41 |
| Target hit | 2023-12-20 05:30:00 | 1890.70 | 1602.35 | 1920.09 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-05-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 05:30:00 | 2842.75 | 2069.01 | 2570.96 | Stage2 pullback-breakout RSI=69 vol=4.6x ATR=98.79 |
| Stop hit — per-position SL triggered | 2024-05-10 05:30:00 | 2694.56 | 2096.12 | 2632.16 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 05:30:00 | 1598.70 | 2023-07-20 05:30:00 | 1622.20 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest1 | 2023-10-06 05:30:00 | 1675.00 | 2023-10-20 05:30:00 | 1649.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest1 | 2023-11-02 05:30:00 | 1716.00 | 2023-11-06 05:30:00 | 1808.41 | PARTIAL | 0.50 | 5.39% |
| BUY | retest1 | 2023-11-02 05:30:00 | 1716.00 | 2023-12-20 05:30:00 | 1890.70 | TARGET_HIT | 0.50 | 10.18% |
| BUY | retest1 | 2024-05-06 05:30:00 | 2842.75 | 2024-05-10 05:30:00 | 2694.56 | STOP_HIT | 1.00 | -5.21% |
