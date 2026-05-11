# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 281.65
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
- **Avg / median % per leg:** 5.62% / 2.99%
- **Sum % (uncompounded):** 39.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 5.62% | 39.3% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 5.62% | 39.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 5.62% | 39.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 00:00:00 | 207.35 | 168.78 | 192.23 | Stage2 pullback-breakout RSI=69 vol=3.3x ATR=7.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 00:00:00 | 223.22 | 172.88 | 205.33 | T1 booked 50% @ 223.22 |
| Stop hit — per-position SL triggered | 2023-07-26 00:00:00 | 213.55 | 176.72 | 212.23 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 00:00:00 | 261.55 | 212.81 | 255.29 | Stage2 pullback-breakout RSI=54 vol=3.9x ATR=9.93 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 246.66 | 214.80 | 254.42 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 277.55 | 217.82 | 255.72 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=11.25 |
| Stop hit — per-position SL triggered | 2023-11-17 00:00:00 | 268.25 | 223.08 | 266.36 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 282.15 | 227.11 | 266.14 | Stage2 pullback-breakout RSI=64 vol=5.7x ATR=9.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 00:00:00 | 302.08 | 230.80 | 277.91 | T1 booked 50% @ 302.08 |
| Target hit | 2024-02-09 00:00:00 | 384.90 | 275.52 | 388.68 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 349.95 | 303.29 | 336.49 | Stage2 pullback-breakout RSI=57 vol=3.6x ATR=13.40 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 329.85 | 305.14 | 337.77 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 00:00:00 | 207.35 | 2023-07-13 00:00:00 | 223.22 | PARTIAL | 0.50 | 7.65% |
| BUY | retest1 | 2023-06-28 00:00:00 | 207.35 | 2023-07-26 00:00:00 | 213.55 | STOP_HIT | 0.50 | 2.99% |
| BUY | retest1 | 2023-10-16 00:00:00 | 261.55 | 2023-10-23 00:00:00 | 246.66 | STOP_HIT | 1.00 | -5.69% |
| BUY | retest1 | 2023-11-03 00:00:00 | 277.55 | 2023-11-17 00:00:00 | 268.25 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest1 | 2023-12-04 00:00:00 | 282.15 | 2023-12-12 00:00:00 | 302.08 | PARTIAL | 0.50 | 7.06% |
| BUY | retest1 | 2023-12-04 00:00:00 | 282.15 | 2024-02-09 00:00:00 | 384.90 | TARGET_HIT | 0.50 | 36.42% |
| BUY | retest1 | 2024-04-29 00:00:00 | 349.95 | 2024-05-07 00:00:00 | 329.85 | STOP_HIT | 1.00 | -5.74% |
