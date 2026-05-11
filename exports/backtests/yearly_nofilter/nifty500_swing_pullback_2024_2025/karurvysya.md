# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 304.65
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 0.46% / 2.70%
- **Sum % (uncompounded):** 2.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 4 | 1 | 0.46% | 2.7% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 4 | 1 | 0.46% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 4 | 1 | 0.46% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 00:00:00 | 171.93 | 149.08 | 168.32 | Stage2 pullback-breakout RSI=55 vol=2.8x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 00:00:00 | 183.25 | 150.77 | 173.02 | T1 booked 50% @ 183.25 |
| Target hit | 2024-08-05 00:00:00 | 176.58 | 152.89 | 179.02 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-09-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 00:00:00 | 183.23 | 161.31 | 181.13 | Stage2 pullback-breakout RSI=53 vol=3.3x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-10-03 00:00:00 | 175.27 | 162.60 | 180.53 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 00:00:00 | 180.76 | 163.23 | 173.78 | Stage2 pullback-breakout RSI=59 vol=6.0x ATR=6.02 |
| Stop hit — per-position SL triggered | 2024-11-04 00:00:00 | 185.95 | 165.61 | 181.60 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 193.05 | 166.32 | 183.73 | Stage2 pullback-breakout RSI=65 vol=2.2x ATR=6.36 |
| Stop hit — per-position SL triggered | 2024-11-12 00:00:00 | 183.50 | 166.97 | 184.85 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-11-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 00:00:00 | 198.65 | 168.55 | 185.21 | Stage2 pullback-breakout RSI=66 vol=3.6x ATR=6.75 |
| Stop hit — per-position SL triggered | 2024-12-12 00:00:00 | 198.38 | 171.51 | 194.46 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-18 00:00:00 | 171.93 | 2024-07-26 00:00:00 | 183.25 | PARTIAL | 0.50 | 6.58% |
| BUY | retest1 | 2024-07-18 00:00:00 | 171.93 | 2024-08-05 00:00:00 | 176.58 | TARGET_HIT | 0.50 | 2.70% |
| BUY | retest1 | 2024-09-23 00:00:00 | 183.23 | 2024-10-03 00:00:00 | 175.27 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest1 | 2024-10-17 00:00:00 | 180.76 | 2024-11-04 00:00:00 | 185.95 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest1 | 2024-11-07 00:00:00 | 193.05 | 2024-11-12 00:00:00 | 183.50 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest1 | 2024-11-28 00:00:00 | 198.65 | 2024-12-12 00:00:00 | 198.38 | STOP_HIT | 1.00 | -0.14% |
