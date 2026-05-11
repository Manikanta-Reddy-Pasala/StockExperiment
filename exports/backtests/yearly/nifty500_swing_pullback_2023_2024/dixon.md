# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 10803.00
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 4
- **Avg / median % per leg:** 4.85% / 5.66%
- **Sum % (uncompounded):** 38.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 1 | 3 | 4 | 4.85% | 38.8% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 1 | 3 | 4 | 4.85% | 38.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 1 | 3 | 4 | 4.85% | 38.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 05:30:00 | 4443.35 | 3760.49 | 4203.23 | Stage2 pullback-breakout RSI=64 vol=5.0x ATR=140.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 05:30:00 | 4723.64 | 3769.01 | 4242.57 | T1 booked 50% @ 4723.64 |
| Stop hit — per-position SL triggered | 2023-08-07 05:30:00 | 4443.35 | 3778.18 | 4285.21 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-09-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 05:30:00 | 5098.15 | 4115.74 | 4952.83 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=144.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 05:30:00 | 5386.92 | 4253.00 | 5170.71 | T1 booked 50% @ 5386.92 |
| Stop hit — per-position SL triggered | 2023-10-30 05:30:00 | 5098.15 | 4352.05 | 5295.27 | SL hit (bars_held=22) |

### Cycle 3 — BUY (started 2023-11-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 05:30:00 | 5510.60 | 4544.55 | 5351.65 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=125.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 05:30:00 | 5762.38 | 4558.55 | 5408.79 | T1 booked 50% @ 5762.38 |
| Target hit | 2024-01-15 05:30:00 | 6331.60 | 5018.86 | 6340.03 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 05:30:00 | 8145.75 | 5978.11 | 7595.84 | Stage2 pullback-breakout RSI=70 vol=1.8x ATR=237.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 05:30:00 | 8621.72 | 6049.18 | 7802.75 | T1 booked 50% @ 8621.72 |
| Stop hit — per-position SL triggered | 2024-05-09 05:30:00 | 8268.60 | 6206.17 | 8086.50 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-03 05:30:00 | 4443.35 | 2023-08-04 05:30:00 | 4723.64 | PARTIAL | 0.50 | 6.31% |
| BUY | retest1 | 2023-08-03 05:30:00 | 4443.35 | 2023-08-07 05:30:00 | 4443.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-26 05:30:00 | 5098.15 | 2023-10-16 05:30:00 | 5386.92 | PARTIAL | 0.50 | 5.66% |
| BUY | retest1 | 2023-09-26 05:30:00 | 5098.15 | 2023-10-30 05:30:00 | 5098.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 05:30:00 | 5510.60 | 2023-12-01 05:30:00 | 5762.38 | PARTIAL | 0.50 | 4.57% |
| BUY | retest1 | 2023-11-30 05:30:00 | 5510.60 | 2024-01-15 05:30:00 | 6331.60 | TARGET_HIT | 0.50 | 14.90% |
| BUY | retest1 | 2024-04-24 05:30:00 | 8145.75 | 2024-04-29 05:30:00 | 8621.72 | PARTIAL | 0.50 | 5.84% |
| BUY | retest1 | 2024-04-24 05:30:00 | 8145.75 | 2024-05-09 05:30:00 | 8268.60 | STOP_HIT | 0.50 | 1.51% |
