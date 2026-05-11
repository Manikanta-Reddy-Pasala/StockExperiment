# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1791.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 1.00% / 0.00%
- **Sum % (uncompounded):** 7.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 1.00% | 8.0% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 1.00% | 8.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 5 | 2 | 1.00% | 8.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 00:00:00 | 946.30 | 687.11 | 888.77 | Stage2 pullback-breakout RSI=68 vol=1.7x ATR=32.75 |
| Stop hit — per-position SL triggered | 2023-07-18 00:00:00 | 917.10 | 710.13 | 913.30 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 00:00:00 | 974.80 | 729.09 | 924.84 | Stage2 pullback-breakout RSI=69 vol=2.5x ATR=30.94 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 928.40 | 733.39 | 928.61 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-08-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 00:00:00 | 963.25 | 765.12 | 926.81 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=29.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 00:00:00 | 1022.41 | 778.31 | 957.06 | T1 booked 50% @ 1022.41 |
| Target hit | 2023-10-04 00:00:00 | 1107.30 | 838.48 | 1116.57 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 1085.60 | 956.78 | 1030.79 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=32.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 00:00:00 | 1150.96 | 962.02 | 1057.97 | T1 booked 50% @ 1150.96 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 1085.60 | 978.32 | 1111.59 | SL hit (bars_held=12) |

### Cycle 5 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 1087.95 | 987.34 | 1084.85 | Stage2 pullback-breakout RSI=51 vol=1.5x ATR=38.49 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 1030.21 | 992.91 | 1070.09 | SL hit (bars_held=8) |

### Cycle 6 — BUY (started 2024-05-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 00:00:00 | 1143.05 | 1020.91 | 1093.01 | Stage2 pullback-breakout RSI=60 vol=4.1x ATR=45.58 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 1074.69 | 1025.20 | 1098.17 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-04 00:00:00 | 946.30 | 2023-07-18 00:00:00 | 917.10 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest1 | 2023-07-31 00:00:00 | 974.80 | 2023-08-02 00:00:00 | 928.40 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest1 | 2023-08-29 00:00:00 | 963.25 | 2023-09-06 00:00:00 | 1022.41 | PARTIAL | 0.50 | 6.14% |
| BUY | retest1 | 2023-08-29 00:00:00 | 963.25 | 2023-10-04 00:00:00 | 1107.30 | TARGET_HIT | 0.50 | 14.95% |
| BUY | retest1 | 2024-01-24 00:00:00 | 1085.60 | 2024-01-30 00:00:00 | 1150.96 | PARTIAL | 0.50 | 6.02% |
| BUY | retest1 | 2024-01-24 00:00:00 | 1085.60 | 2024-02-12 00:00:00 | 1085.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-26 00:00:00 | 1087.95 | 2024-03-06 00:00:00 | 1030.21 | STOP_HIT | 1.00 | -5.31% |
| BUY | retest1 | 2024-05-02 00:00:00 | 1143.05 | 2024-05-09 00:00:00 | 1074.69 | STOP_HIT | 1.00 | -5.98% |
