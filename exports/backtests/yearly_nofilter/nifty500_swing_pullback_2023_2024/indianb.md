# Indian Bank (INDIANB)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 3.74% / 0.00%
- **Sum % (uncompounded):** 26.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 3.74% | 26.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 3.74% | 26.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 3.74% | 26.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 292.55 | 269.31 | 282.33 | Stage2 pullback-breakout RSI=60 vol=3.1x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 305.92 | 269.90 | 285.38 | T1 booked 50% @ 305.92 |
| Target hit | 2023-08-30 00:00:00 | 380.55 | 299.32 | 383.18 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 00:00:00 | 412.15 | 310.59 | 391.35 | Stage2 pullback-breakout RSI=65 vol=4.3x ATR=14.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 00:00:00 | 441.28 | 317.08 | 405.78 | T1 booked 50% @ 441.28 |
| Stop hit — per-position SL triggered | 2023-09-28 00:00:00 | 412.15 | 318.05 | 406.55 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 447.05 | 345.74 | 422.38 | Stage2 pullback-breakout RSI=63 vol=3.0x ATR=14.09 |
| Stop hit — per-position SL triggered | 2023-11-20 00:00:00 | 425.92 | 349.30 | 426.72 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-03-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-06 00:00:00 | 538.65 | 415.88 | 524.09 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=20.66 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 507.65 | 419.45 | 527.12 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 559.05 | 444.55 | 523.31 | Stage2 pullback-breakout RSI=65 vol=2.7x ATR=18.98 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 530.58 | 448.42 | 529.73 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 292.55 | 2023-07-04 00:00:00 | 305.92 | PARTIAL | 0.50 | 4.57% |
| BUY | retest1 | 2023-06-30 00:00:00 | 292.55 | 2023-08-30 00:00:00 | 380.55 | TARGET_HIT | 0.50 | 30.08% |
| BUY | retest1 | 2023-09-18 00:00:00 | 412.15 | 2023-09-27 00:00:00 | 441.28 | PARTIAL | 0.50 | 7.07% |
| BUY | retest1 | 2023-09-18 00:00:00 | 412.15 | 2023-09-28 00:00:00 | 412.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-13 00:00:00 | 447.05 | 2023-11-20 00:00:00 | 425.92 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest1 | 2024-03-06 00:00:00 | 538.65 | 2024-03-12 00:00:00 | 507.65 | STOP_HIT | 1.00 | -5.75% |
| BUY | retest1 | 2024-04-29 00:00:00 | 559.05 | 2024-05-06 00:00:00 | 530.58 | STOP_HIT | 1.00 | -5.09% |
