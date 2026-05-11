# Adani Energy Solutions Ltd. (ADANIENSOL)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1353.60
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 1.61% / 2.43%
- **Sum % (uncompounded):** 11.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 0 | 5 | 2 | 1.61% | 11.2% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 0 | 5 | 2 | 1.61% | 11.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 0 | 5 | 2 | 1.61% | 11.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 05:30:00 | 967.55 | 865.79 | 925.95 | Stage2 pullback-breakout RSI=64 vol=3.8x ATR=24.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 05:30:00 | 1017.32 | 875.54 | 957.96 | T1 booked 50% @ 1017.32 |
| Stop hit — per-position SL triggered | 2025-11-20 05:30:00 | 993.60 | 883.69 | 983.22 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-12-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 05:30:00 | 982.55 | 896.08 | 979.07 | Stage2 pullback-breakout RSI=52 vol=1.6x ATR=25.51 |
| Stop hit — per-position SL triggered | 2025-12-24 05:30:00 | 996.55 | 905.60 | 989.44 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 05:30:00 | 1046.40 | 911.31 | 1003.19 | Stage2 pullback-breakout RSI=65 vol=2.6x ATR=24.99 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 1008.91 | 917.33 | 1014.58 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 974.40 | 914.98 | 919.77 | Stage2 pullback-breakout RSI=55 vol=2.6x ATR=44.45 |
| Stop hit — per-position SL triggered | 2026-02-20 05:30:00 | 998.10 | 927.09 | 988.44 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-03-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 05:30:00 | 1013.60 | 934.63 | 994.15 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=35.54 |
| Stop hit — per-position SL triggered | 2026-03-23 05:30:00 | 960.29 | 940.10 | 995.72 | SL hit (bars_held=9) |

### Cycle 6 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 1073.30 | 943.36 | 988.58 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=45.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 05:30:00 | 1163.44 | 949.11 | 1027.99 | T1 booked 50% @ 1163.44 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-29 05:30:00 | 967.55 | 2025-11-12 05:30:00 | 1017.32 | PARTIAL | 0.50 | 5.14% |
| BUY | retest1 | 2025-10-29 05:30:00 | 967.55 | 2025-11-20 05:30:00 | 993.60 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest1 | 2025-12-10 05:30:00 | 982.55 | 2025-12-24 05:30:00 | 996.55 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest1 | 2026-01-01 05:30:00 | 1046.40 | 2026-01-08 05:30:00 | 1008.91 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest1 | 2026-02-03 05:30:00 | 974.40 | 2026-02-20 05:30:00 | 998.10 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest1 | 2026-03-10 05:30:00 | 1013.60 | 2026-03-23 05:30:00 | 960.29 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest1 | 2026-04-08 05:30:00 | 1073.30 | 2026-04-13 05:30:00 | 1163.44 | PARTIAL | 0.50 | 8.40% |
