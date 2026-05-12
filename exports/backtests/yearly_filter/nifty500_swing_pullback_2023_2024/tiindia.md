# Tube Investments of India Ltd. (TIINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3007.50
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 4
- **Avg / median % per leg:** 3.54% / 5.84%
- **Sum % (uncompounded):** 28.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 1 | 3 | 4 | 3.54% | 28.3% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 3 | 4 | 3.54% | 28.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 1 | 3 | 4 | 3.54% | 28.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 2988.15 | 2819.11 | 2905.70 | Stage2 pullback-breakout RSI=56 vol=1.9x ATR=87.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 00:00:00 | 3162.77 | 2824.92 | 2943.70 | T1 booked 50% @ 3162.77 |
| Target hit | 2023-09-22 00:00:00 | 3200.30 | 2881.58 | 3207.37 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 00:00:00 | 3574.20 | 3001.60 | 3336.32 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=131.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 00:00:00 | 3836.73 | 3010.50 | 3389.61 | T1 booked 50% @ 3836.73 |
| Stop hit — per-position SL triggered | 2023-12-19 00:00:00 | 3574.20 | 3051.20 | 3528.31 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-01-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 00:00:00 | 3751.05 | 3112.37 | 3577.56 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=134.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 00:00:00 | 4019.92 | 3141.39 | 3670.25 | T1 booked 50% @ 4019.92 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 3751.05 | 3178.31 | 3758.14 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 3707.10 | 3343.34 | 3581.99 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=132.07 |
| Stop hit — per-position SL triggered | 2024-04-08 00:00:00 | 3549.75 | 3376.92 | 3643.02 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 3707.60 | 3399.20 | 3594.63 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=95.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 00:00:00 | 3897.91 | 3416.59 | 3679.63 | T1 booked 50% @ 3897.91 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-01 00:00:00 | 2988.15 | 2023-09-05 00:00:00 | 3162.77 | PARTIAL | 0.50 | 5.84% |
| BUY | retest1 | 2023-09-01 00:00:00 | 2988.15 | 2023-09-22 00:00:00 | 3200.30 | TARGET_HIT | 0.50 | 7.10% |
| BUY | retest1 | 2023-12-08 00:00:00 | 3574.20 | 2023-12-11 00:00:00 | 3836.73 | PARTIAL | 0.50 | 7.35% |
| BUY | retest1 | 2023-12-08 00:00:00 | 3574.20 | 2023-12-19 00:00:00 | 3574.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-05 00:00:00 | 3751.05 | 2024-01-11 00:00:00 | 4019.92 | PARTIAL | 0.50 | 7.17% |
| BUY | retest1 | 2024-01-05 00:00:00 | 3751.05 | 2024-01-18 00:00:00 | 3751.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 00:00:00 | 3707.10 | 2024-04-08 00:00:00 | 3549.75 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest1 | 2024-04-29 00:00:00 | 3707.60 | 2024-05-06 00:00:00 | 3897.91 | PARTIAL | 0.50 | 5.13% |
