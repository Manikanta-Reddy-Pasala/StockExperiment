# Indian Bank (INDIANB)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 865.55
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
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.49% / 0.00%
- **Sum % (uncompounded):** 3.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.49% | 3.4% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.49% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.49% | 3.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 00:00:00 | 578.65 | 495.80 | 546.81 | Stage2 pullback-breakout RSI=66 vol=2.3x ATR=16.56 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 553.80 | 499.30 | 554.54 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 587.25 | 502.33 | 562.06 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=18.47 |
| Stop hit — per-position SL triggered | 2024-08-06 00:00:00 | 559.55 | 507.50 | 574.26 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-08-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 00:00:00 | 567.95 | 514.22 | 559.11 | Stage2 pullback-breakout RSI=54 vol=1.6x ATR=18.01 |
| Stop hit — per-position SL triggered | 2024-09-04 00:00:00 | 540.93 | 516.16 | 556.61 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-10-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 00:00:00 | 551.45 | 517.51 | 518.73 | Stage2 pullback-breakout RSI=63 vol=5.8x ATR=18.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 00:00:00 | 588.59 | 518.84 | 530.70 | T1 booked 50% @ 588.59 |
| Stop hit — per-position SL triggered | 2024-11-11 00:00:00 | 551.45 | 523.18 | 554.37 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 564.85 | 524.45 | 546.33 | Stage2 pullback-breakout RSI=58 vol=3.3x ATR=23.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 00:00:00 | 610.99 | 527.99 | 562.82 | T1 booked 50% @ 610.99 |
| Stop hit — per-position SL triggered | 2024-12-12 00:00:00 | 577.65 | 531.68 | 575.22 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-15 00:00:00 | 578.65 | 2024-07-23 00:00:00 | 553.80 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest1 | 2024-07-29 00:00:00 | 587.25 | 2024-08-06 00:00:00 | 559.55 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest1 | 2024-08-28 00:00:00 | 567.95 | 2024-09-04 00:00:00 | 540.93 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest1 | 2024-10-28 00:00:00 | 551.45 | 2024-10-30 00:00:00 | 588.59 | PARTIAL | 0.50 | 6.74% |
| BUY | retest1 | 2024-10-28 00:00:00 | 551.45 | 2024-11-11 00:00:00 | 551.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 00:00:00 | 564.85 | 2024-12-04 00:00:00 | 610.99 | PARTIAL | 0.50 | 8.17% |
| BUY | retest1 | 2024-11-25 00:00:00 | 564.85 | 2024-12-12 00:00:00 | 577.65 | STOP_HIT | 0.50 | 2.27% |
