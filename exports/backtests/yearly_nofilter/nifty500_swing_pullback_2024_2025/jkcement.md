# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 5461.50
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.79% / 2.75%
- **Sum % (uncompounded):** 5.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.79% | 5.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.79% | 5.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.79% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 00:00:00 | 4508.20 | 3923.51 | 4233.60 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=138.62 |
| Stop hit — per-position SL triggered | 2024-07-03 00:00:00 | 4300.27 | 3941.20 | 4278.65 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-07-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 00:00:00 | 4471.95 | 3985.14 | 4325.37 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=128.60 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 4279.04 | 4022.22 | 4374.49 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 00:00:00 | 4401.00 | 4050.50 | 4296.55 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=117.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 00:00:00 | 4635.81 | 4083.14 | 4400.89 | T1 booked 50% @ 4635.81 |
| Target hit | 2024-09-26 00:00:00 | 4582.20 | 4178.99 | 4636.00 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-12-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 00:00:00 | 4495.65 | 4199.96 | 4187.51 | Stage2 pullback-breakout RSI=66 vol=3.4x ATR=126.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 00:00:00 | 4749.16 | 4228.43 | 4406.45 | T1 booked 50% @ 4749.16 |
| Stop hit — per-position SL triggered | 2024-12-18 00:00:00 | 4619.15 | 4250.89 | 4516.56 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 00:00:00 | 4745.35 | 4283.33 | 4570.35 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=107.97 |
| Stop hit — per-position SL triggered | 2025-01-10 00:00:00 | 4583.39 | 4308.40 | 4631.82 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-27 00:00:00 | 4508.20 | 2024-07-03 00:00:00 | 4300.27 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest1 | 2024-07-22 00:00:00 | 4471.95 | 2024-08-02 00:00:00 | 4279.04 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest1 | 2024-08-22 00:00:00 | 4401.00 | 2024-09-03 00:00:00 | 4635.81 | PARTIAL | 0.50 | 5.34% |
| BUY | retest1 | 2024-08-22 00:00:00 | 4401.00 | 2024-09-26 00:00:00 | 4582.20 | TARGET_HIT | 0.50 | 4.12% |
| BUY | retest1 | 2024-12-02 00:00:00 | 4495.65 | 2024-12-11 00:00:00 | 4749.16 | PARTIAL | 0.50 | 5.64% |
| BUY | retest1 | 2024-12-02 00:00:00 | 4495.65 | 2024-12-18 00:00:00 | 4619.15 | STOP_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2025-01-02 00:00:00 | 4745.35 | 2025-01-10 00:00:00 | 4583.39 | STOP_HIT | 1.00 | -3.41% |
