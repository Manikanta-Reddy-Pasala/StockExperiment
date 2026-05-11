# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 3583.10
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
- **Avg / median % per leg:** 1.16% / 2.20%
- **Sum % (uncompounded):** 5.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | 1.16% | 5.8% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 1.16% | 5.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | 1.16% | 5.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 00:00:00 | 2047.20 | 1615.83 | 1974.43 | Stage2 pullback-breakout RSI=62 vol=2.2x ATR=54.20 |
| Stop hit — per-position SL triggered | 2024-07-12 00:00:00 | 1965.90 | 1632.73 | 1997.67 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 00:00:00 | 1906.90 | 1694.13 | 1851.74 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=60.98 |
| Stop hit — per-position SL triggered | 2024-09-13 00:00:00 | 1959.50 | 1718.09 | 1912.24 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 00:00:00 | 2063.45 | 1756.05 | 1957.29 | Stage2 pullback-breakout RSI=68 vol=2.8x ATR=66.14 |
| Stop hit — per-position SL triggered | 2024-10-23 00:00:00 | 2023.38 | 1784.01 | 2016.45 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 2062.60 | 1823.62 | 1996.32 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=69.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 00:00:00 | 2202.15 | 1843.97 | 2051.46 | T1 booked 50% @ 2202.15 |
| Target hit | 2024-12-17 00:00:00 | 2107.98 | 1869.27 | 2114.55 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 00:00:00 | 2047.20 | 2024-07-12 00:00:00 | 1965.90 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest1 | 2024-08-30 00:00:00 | 1906.90 | 2024-09-13 00:00:00 | 1959.50 | STOP_HIT | 1.00 | 2.76% |
| BUY | retest1 | 2024-10-09 00:00:00 | 2063.45 | 2024-10-23 00:00:00 | 2023.38 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest1 | 2024-11-25 00:00:00 | 2062.60 | 2024-12-05 00:00:00 | 2202.15 | PARTIAL | 0.50 | 6.77% |
| BUY | retest1 | 2024-11-25 00:00:00 | 2062.60 | 2024-12-17 00:00:00 | 2107.98 | TARGET_HIT | 0.50 | 2.20% |
