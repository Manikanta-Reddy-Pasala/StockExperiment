# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1569.80
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 2.08% / 2.98%
- **Sum % (uncompounded):** 12.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 0 | 4 | 2 | 2.08% | 12.5% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 0 | 4 | 2 | 2.08% | 12.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 0 | 4 | 2 | 2.08% | 12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 00:00:00 | 1908.70 | 1651.39 | 1870.14 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=45.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 00:00:00 | 1999.21 | 1674.40 | 1903.13 | T1 booked 50% @ 1999.21 |
| Stop hit — per-position SL triggered | 2023-08-08 00:00:00 | 1965.50 | 1689.88 | 1937.70 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 1991.00 | 1810.00 | 1925.36 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=37.70 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 1934.45 | 1815.83 | 1935.91 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 1868.75 | 1828.61 | 1833.51 | Stage2 pullback-breakout RSI=54 vol=1.9x ATR=47.03 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 1899.70 | 1834.38 | 1869.40 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 2075.90 | 1844.89 | 1928.17 | Stage2 pullback-breakout RSI=69 vol=3.7x ATR=54.72 |
| Stop hit — per-position SL triggered | 2024-03-07 00:00:00 | 2100.25 | 1868.06 | 2030.96 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 2010.65 | 1901.33 | 1991.75 | Stage2 pullback-breakout RSI=53 vol=1.7x ATR=48.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 00:00:00 | 2106.94 | 1906.35 | 2013.02 | T1 booked 50% @ 2106.94 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-19 00:00:00 | 1908.70 | 2023-08-01 00:00:00 | 1999.21 | PARTIAL | 0.50 | 4.74% |
| BUY | retest1 | 2023-07-19 00:00:00 | 1908.70 | 2023-08-08 00:00:00 | 1965.50 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest1 | 2023-12-04 00:00:00 | 1991.00 | 2023-12-08 00:00:00 | 1934.45 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2024-01-29 00:00:00 | 1868.75 | 2024-02-13 00:00:00 | 1899.70 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest1 | 2024-02-23 00:00:00 | 2075.90 | 2024-03-07 00:00:00 | 2100.25 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest1 | 2024-04-25 00:00:00 | 2010.65 | 2024-04-30 00:00:00 | 2106.94 | PARTIAL | 0.50 | 4.79% |
