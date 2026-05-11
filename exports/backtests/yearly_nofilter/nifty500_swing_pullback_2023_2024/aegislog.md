# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 722.60
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 4.01% / 5.28%
- **Sum % (uncompounded):** 28.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 3 | 3 | 4.01% | 28.1% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 3 | 3 | 4.01% | 28.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 3 | 3 | 4.01% | 28.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 00:00:00 | 353.05 | 343.71 | 337.71 | Stage2 pullback-breakout RSI=57 vol=6.1x ATR=12.27 |
| Stop hit — per-position SL triggered | 2023-07-19 00:00:00 | 361.55 | 344.70 | 348.65 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 00:00:00 | 378.30 | 349.88 | 365.37 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=12.82 |
| Stop hit — per-position SL triggered | 2023-09-04 00:00:00 | 359.07 | 351.34 | 366.98 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2024-01-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 00:00:00 | 355.10 | 343.74 | 352.22 | Stage2 pullback-breakout RSI=52 vol=2.3x ATR=11.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 00:00:00 | 377.58 | 345.16 | 359.67 | T1 booked 50% @ 377.58 |
| Target hit | 2024-02-09 00:00:00 | 373.85 | 350.18 | 376.23 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 00:00:00 | 426.70 | 351.97 | 383.41 | Stage2 pullback-breakout RSI=66 vol=4.1x ATR=19.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 00:00:00 | 464.92 | 352.81 | 388.43 | T1 booked 50% @ 464.92 |
| Stop hit — per-position SL triggered | 2024-02-26 00:00:00 | 426.70 | 358.50 | 415.50 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 446.65 | 368.60 | 405.41 | Stage2 pullback-breakout RSI=62 vol=8.7x ATR=22.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 00:00:00 | 492.17 | 376.18 | 435.42 | T1 booked 50% @ 492.17 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-05 00:00:00 | 353.05 | 2023-07-19 00:00:00 | 361.55 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest1 | 2023-08-23 00:00:00 | 378.30 | 2023-09-04 00:00:00 | 359.07 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest1 | 2024-01-10 00:00:00 | 355.10 | 2024-01-18 00:00:00 | 377.58 | PARTIAL | 0.50 | 6.33% |
| BUY | retest1 | 2024-01-10 00:00:00 | 355.10 | 2024-02-09 00:00:00 | 373.85 | TARGET_HIT | 0.50 | 5.28% |
| BUY | retest1 | 2024-02-15 00:00:00 | 426.70 | 2024-02-16 00:00:00 | 464.92 | PARTIAL | 0.50 | 8.96% |
| BUY | retest1 | 2024-02-15 00:00:00 | 426.70 | 2024-02-26 00:00:00 | 426.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-28 00:00:00 | 446.65 | 2024-04-15 00:00:00 | 492.17 | PARTIAL | 0.50 | 10.19% |
