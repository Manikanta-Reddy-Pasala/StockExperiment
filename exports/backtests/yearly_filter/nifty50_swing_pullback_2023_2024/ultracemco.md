# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 11950.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** 0.45% / -2.55%
- **Sum % (uncompounded):** 3.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 5 | 1 | 0.45% | 3.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 5 | 1 | 0.45% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 5 | 1 | 0.45% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 00:00:00 | 8393.30 | 7491.67 | 8249.02 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=146.95 |
| Stop hit — per-position SL triggered | 2023-08-03 00:00:00 | 8172.87 | 7545.01 | 8260.38 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 8582.55 | 7669.40 | 8241.87 | Stage2 pullback-breakout RSI=68 vol=2.2x ATR=135.53 |
| Stop hit — per-position SL triggered | 2023-09-18 00:00:00 | 8628.00 | 7754.67 | 8459.61 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 00:00:00 | 8518.55 | 7851.85 | 8311.05 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=144.91 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 8301.18 | 7866.48 | 8318.76 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 9003.65 | 8034.89 | 8668.82 | Stage2 pullback-breakout RSI=67 vol=4.4x ATR=138.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 9281.06 | 8057.49 | 8762.08 | T1 booked 50% @ 9281.06 |
| Target hit | 2024-01-08 00:00:00 | 9934.50 | 8462.45 | 9950.56 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 10093.70 | 8586.95 | 9933.07 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=191.09 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 9807.06 | 8626.93 | 9934.76 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 9964.45 | 9171.72 | 9685.73 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=184.87 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 9687.15 | 9204.53 | 9742.71 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-25 00:00:00 | 8393.30 | 2023-08-03 00:00:00 | 8172.87 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest1 | 2023-09-04 00:00:00 | 8582.55 | 2023-09-18 00:00:00 | 8628.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest1 | 2023-10-19 00:00:00 | 8518.55 | 2023-10-25 00:00:00 | 8301.18 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest1 | 2023-11-30 00:00:00 | 9003.65 | 2023-12-04 00:00:00 | 9281.06 | PARTIAL | 0.50 | 3.08% |
| BUY | retest1 | 2023-11-30 00:00:00 | 9003.65 | 2024-01-08 00:00:00 | 9934.50 | TARGET_HIT | 0.50 | 10.34% |
| BUY | retest1 | 2024-01-19 00:00:00 | 10093.70 | 2024-01-24 00:00:00 | 9807.06 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2024-04-29 00:00:00 | 9964.45 | 2024-05-07 00:00:00 | 9687.15 | STOP_HIT | 1.00 | -2.78% |
