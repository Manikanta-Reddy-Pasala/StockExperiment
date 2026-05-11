# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 7012.50
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
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 4.79% / 4.40%
- **Sum % (uncompounded):** 33.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 4.79% | 33.6% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 4.79% | 33.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 4.79% | 33.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 05:30:00 | 4500.30 | 3820.90 | 4366.42 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=99.68 |
| Stop hit — per-position SL triggered | 2023-09-13 05:30:00 | 4350.78 | 3851.74 | 4398.49 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-11-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 05:30:00 | 4310.50 | 3946.98 | 4153.57 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=101.09 |
| Stop hit — per-position SL triggered | 2023-11-23 05:30:00 | 4260.55 | 3977.77 | 4229.74 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 05:30:00 | 4419.75 | 3990.24 | 4253.36 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=100.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 05:30:00 | 4620.35 | 4000.53 | 4300.84 | T1 booked 50% @ 4620.35 |
| Target hit | 2023-12-20 05:30:00 | 4614.35 | 4088.30 | 4632.65 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 05:30:00 | 4920.40 | 4169.23 | 4731.91 | Stage2 pullback-breakout RSI=64 vol=2.5x ATR=127.21 |
| Stop hit — per-position SL triggered | 2024-01-15 05:30:00 | 4729.58 | 4195.35 | 4763.68 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-02-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 05:30:00 | 4984.70 | 4290.61 | 4595.50 | Stage2 pullback-breakout RSI=66 vol=9.6x ATR=167.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-22 05:30:00 | 5320.38 | 4301.90 | 4674.50 | T1 booked 50% @ 5320.38 |
| Target hit | 2024-04-19 05:30:00 | 6292.60 | 4839.54 | 6318.24 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-06 05:30:00 | 4500.30 | 2023-09-13 05:30:00 | 4350.78 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest1 | 2023-11-09 05:30:00 | 4310.50 | 2023-11-23 05:30:00 | 4260.55 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest1 | 2023-11-30 05:30:00 | 4419.75 | 2023-12-04 05:30:00 | 4620.35 | PARTIAL | 0.50 | 4.54% |
| BUY | retest1 | 2023-11-30 05:30:00 | 4419.75 | 2023-12-20 05:30:00 | 4614.35 | TARGET_HIT | 0.50 | 4.40% |
| BUY | retest1 | 2024-01-09 05:30:00 | 4920.40 | 2024-01-15 05:30:00 | 4729.58 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest1 | 2024-02-21 05:30:00 | 4984.70 | 2024-02-22 05:30:00 | 5320.38 | PARTIAL | 0.50 | 6.73% |
| BUY | retest1 | 2024-02-21 05:30:00 | 4984.70 | 2024-04-19 05:30:00 | 6292.60 | TARGET_HIT | 0.50 | 26.24% |
