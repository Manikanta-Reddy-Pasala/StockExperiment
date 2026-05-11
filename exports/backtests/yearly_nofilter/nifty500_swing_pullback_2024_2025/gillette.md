# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 8017.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 0.75% / 0.00%
- **Sum % (uncompounded):** 7.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.75% | 7.5% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.75% | 7.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.75% | 7.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 00:00:00 | 7675.90 | 6676.87 | 7393.80 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=203.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 00:00:00 | 8083.81 | 6700.18 | 7478.28 | T1 booked 50% @ 8083.81 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 7675.90 | 6709.97 | 7497.86 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-08-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 00:00:00 | 7981.90 | 6815.93 | 7716.54 | Stage2 pullback-breakout RSI=66 vol=2.7x ATR=216.20 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 7657.59 | 6823.69 | 7705.00 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2024-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 00:00:00 | 8266.60 | 6936.65 | 7858.91 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=221.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 00:00:00 | 8710.19 | 6985.52 | 8053.55 | T1 booked 50% @ 8710.19 |
| Target hit | 2024-09-23 00:00:00 | 8817.15 | 7342.32 | 8824.72 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-10-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 00:00:00 | 8817.80 | 7460.10 | 8676.21 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=260.23 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 8427.45 | 7595.84 | 8778.42 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2024-10-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 00:00:00 | 9402.50 | 7644.22 | 8718.80 | Stage2 pullback-breakout RSI=65 vol=12.8x ATR=379.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 00:00:00 | 10161.96 | 7737.33 | 9154.94 | T1 booked 50% @ 10161.96 |
| Stop hit — per-position SL triggered | 2024-11-12 00:00:00 | 9402.50 | 7864.56 | 9477.28 | SL hit (bars_held=10) |

### Cycle 6 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 10443.15 | 7977.08 | 9519.47 | Stage2 pullback-breakout RSI=66 vol=2.5x ATR=441.17 |
| Stop hit — per-position SL triggered | 2024-11-28 00:00:00 | 9781.39 | 8033.79 | 9614.92 | SL hit (bars_held=3) |

### Cycle 7 — BUY (started 2024-12-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 00:00:00 | 9992.65 | 8121.58 | 9704.13 | Stage2 pullback-breakout RSI=58 vol=2.4x ATR=409.47 |
| Stop hit — per-position SL triggered | 2024-12-19 00:00:00 | 9687.25 | 8283.62 | 9766.24 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-15 00:00:00 | 7675.90 | 2024-07-18 00:00:00 | 8083.81 | PARTIAL | 0.50 | 5.31% |
| BUY | retest1 | 2024-07-15 00:00:00 | 7675.90 | 2024-07-19 00:00:00 | 7675.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-02 00:00:00 | 7981.90 | 2024-08-05 00:00:00 | 7657.59 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest1 | 2024-08-21 00:00:00 | 8266.60 | 2024-08-26 00:00:00 | 8710.19 | PARTIAL | 0.50 | 5.37% |
| BUY | retest1 | 2024-08-21 00:00:00 | 8266.60 | 2024-09-23 00:00:00 | 8817.15 | TARGET_HIT | 0.50 | 6.66% |
| BUY | retest1 | 2024-10-08 00:00:00 | 8817.80 | 2024-10-22 00:00:00 | 8427.45 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2024-10-29 00:00:00 | 9402.50 | 2024-11-04 00:00:00 | 10161.96 | PARTIAL | 0.50 | 8.08% |
| BUY | retest1 | 2024-10-29 00:00:00 | 9402.50 | 2024-11-12 00:00:00 | 9402.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 00:00:00 | 10443.15 | 2024-11-28 00:00:00 | 9781.39 | STOP_HIT | 1.00 | -6.34% |
| BUY | retest1 | 2024-12-05 00:00:00 | 9992.65 | 2024-12-19 00:00:00 | 9687.25 | STOP_HIT | 1.00 | -3.06% |
