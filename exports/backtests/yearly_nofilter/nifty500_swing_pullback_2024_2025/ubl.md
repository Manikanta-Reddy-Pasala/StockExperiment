# United Breweries Ltd. (UBL)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 1414.40
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 0.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.05% | 0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.05% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.05% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 00:00:00 | 2072.60 | 1814.23 | 2026.28 | Stage2 pullback-breakout RSI=58 vol=3.6x ATR=54.02 |
| Stop hit — per-position SL triggered | 2024-07-22 00:00:00 | 2014.75 | 1838.95 | 2052.79 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 00:00:00 | 2130.50 | 1901.18 | 2056.62 | Stage2 pullback-breakout RSI=63 vol=4.1x ATR=50.31 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 2105.10 | 1924.38 | 2112.97 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-12-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 00:00:00 | 2052.75 | 1941.87 | 1986.96 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=43.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 00:00:00 | 2138.78 | 1948.66 | 2022.00 | T1 booked 50% @ 2138.78 |
| Stop hit — per-position SL triggered | 2025-01-06 00:00:00 | 2052.75 | 1951.69 | 2036.26 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-05 00:00:00 | 2072.60 | 2024-07-22 00:00:00 | 2014.75 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest1 | 2024-09-19 00:00:00 | 2130.50 | 2024-10-04 00:00:00 | 2105.10 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest1 | 2024-12-24 00:00:00 | 2052.75 | 2025-01-02 00:00:00 | 2138.78 | PARTIAL | 0.50 | 4.19% |
| BUY | retest1 | 2024-12-24 00:00:00 | 2052.75 | 2025-01-06 00:00:00 | 2052.75 | STOP_HIT | 0.50 | 0.00% |
