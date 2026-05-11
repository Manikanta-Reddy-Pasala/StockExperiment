# ITI Ltd. (ITI)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 300.05
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
- **Avg / median % per leg:** 2.21% / 2.12%
- **Sum % (uncompounded):** 15.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.21% | 15.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.21% | 15.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.21% | 15.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 05:30:00 | 110.75 | 104.95 | 108.39 | Stage2 pullback-breakout RSI=60 vol=4.3x ATR=2.50 |
| Stop hit — per-position SL triggered | 2023-08-01 05:30:00 | 113.10 | 105.50 | 110.10 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 05:30:00 | 278.75 | 159.79 | 264.13 | Stage2 pullback-breakout RSI=62 vol=3.4x ATR=14.98 |
| Stop hit — per-position SL triggered | 2023-12-05 05:30:00 | 272.10 | 170.67 | 270.13 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 05:30:00 | 300.10 | 172.95 | 273.10 | Stage2 pullback-breakout RSI=68 vol=8.2x ATR=14.32 |
| Stop hit — per-position SL triggered | 2023-12-20 05:30:00 | 278.61 | 183.71 | 287.44 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-01-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 05:30:00 | 315.30 | 202.88 | 303.91 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=13.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 05:30:00 | 341.47 | 204.58 | 310.57 | T1 booked 50% @ 341.47 |
| Target hit | 2024-02-09 05:30:00 | 327.55 | 226.28 | 336.56 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 05:30:00 | 279.45 | 245.10 | 264.74 | Stage2 pullback-breakout RSI=56 vol=9.5x ATR=14.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 05:30:00 | 309.40 | 245.58 | 267.46 | T1 booked 50% @ 309.40 |
| Stop hit — per-position SL triggered | 2024-05-09 05:30:00 | 279.45 | 250.76 | 285.13 | SL hit (bars_held=12) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-18 05:30:00 | 110.75 | 2023-08-01 05:30:00 | 113.10 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest1 | 2023-11-20 05:30:00 | 278.75 | 2023-12-05 05:30:00 | 272.10 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest1 | 2023-12-07 05:30:00 | 300.10 | 2023-12-20 05:30:00 | 278.61 | STOP_HIT | 1.00 | -7.16% |
| BUY | retest1 | 2024-01-15 05:30:00 | 315.30 | 2024-01-16 05:30:00 | 341.47 | PARTIAL | 0.50 | 8.30% |
| BUY | retest1 | 2024-01-15 05:30:00 | 315.30 | 2024-02-09 05:30:00 | 327.55 | TARGET_HIT | 0.50 | 3.89% |
| BUY | retest1 | 2024-04-22 05:30:00 | 279.45 | 2024-04-23 05:30:00 | 309.40 | PARTIAL | 0.50 | 10.72% |
| BUY | retest1 | 2024-04-22 05:30:00 | 279.45 | 2024-05-09 05:30:00 | 279.45 | STOP_HIT | 0.50 | 0.00% |
