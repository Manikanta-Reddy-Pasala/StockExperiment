# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 10478.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 1.58% / 2.28%
- **Sum % (uncompounded):** 11.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 2 | 3 | 2 | 1.58% | 11.0% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 3 | 2 | 1.58% | 11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 2 | 3 | 2 | 1.58% | 11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 00:00:00 | 7139.35 | 6372.96 | 6953.96 | Stage2 pullback-breakout RSI=61 vol=4.1x ATR=159.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 00:00:00 | 7459.02 | 6421.48 | 7072.36 | T1 booked 50% @ 7459.02 |
| Target hit | 2023-08-11 00:00:00 | 7259.65 | 6624.42 | 7396.88 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 7437.70 | 6697.22 | 7291.39 | Stage2 pullback-breakout RSI=59 vol=2.4x ATR=142.48 |
| Stop hit — per-position SL triggered | 2023-09-04 00:00:00 | 7223.98 | 6708.83 | 7289.49 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 7347.80 | 6833.89 | 7055.29 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=147.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 7643.29 | 6888.23 | 7290.63 | T1 booked 50% @ 7643.29 |
| Target hit | 2023-12-21 00:00:00 | 7742.95 | 7015.96 | 7753.93 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 8472.80 | 7197.13 | 8038.78 | Stage2 pullback-breakout RSI=66 vol=3.7x ATR=221.68 |
| Stop hit — per-position SL triggered | 2024-01-23 00:00:00 | 8140.28 | 7216.82 | 8065.92 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 8611.85 | 7324.48 | 8263.28 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=219.50 |
| Stop hit — per-position SL triggered | 2024-02-21 00:00:00 | 8807.80 | 7460.41 | 8583.31 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-04 00:00:00 | 7139.35 | 2023-07-12 00:00:00 | 7459.02 | PARTIAL | 0.50 | 4.48% |
| BUY | retest1 | 2023-07-04 00:00:00 | 7139.35 | 2023-08-11 00:00:00 | 7259.65 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2023-08-31 00:00:00 | 7437.70 | 2023-09-04 00:00:00 | 7223.98 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest1 | 2023-11-17 00:00:00 | 7347.80 | 2023-12-04 00:00:00 | 7643.29 | PARTIAL | 0.50 | 4.02% |
| BUY | retest1 | 2023-11-17 00:00:00 | 7347.80 | 2023-12-21 00:00:00 | 7742.95 | TARGET_HIT | 0.50 | 5.38% |
| BUY | retest1 | 2024-01-19 00:00:00 | 8472.80 | 2024-01-23 00:00:00 | 8140.28 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2024-02-07 00:00:00 | 8611.85 | 2024-02-21 00:00:00 | 8807.80 | STOP_HIT | 1.00 | 2.28% |
