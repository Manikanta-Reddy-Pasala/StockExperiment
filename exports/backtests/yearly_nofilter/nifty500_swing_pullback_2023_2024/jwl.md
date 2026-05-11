# Jupiter Wagons Ltd. (JWL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 290.20
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 2.97% / 1.25%
- **Sum % (uncompounded):** 26.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 1 | 5 | 3 | 2.97% | 26.7% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 5 | 3 | 2.97% | 26.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 5 | 3 | 2.97% | 26.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 00:00:00 | 316.95 | 208.09 | 308.65 | Stage2 pullback-breakout RSI=53 vol=2.2x ATR=17.08 |
| Stop hit — per-position SL triggered | 2023-11-12 00:00:00 | 320.90 | 218.01 | 311.72 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 329.35 | 221.98 | 314.31 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=12.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 00:00:00 | 354.77 | 224.36 | 319.37 | T1 booked 50% @ 354.77 |
| Stop hit — per-position SL triggered | 2023-11-23 00:00:00 | 329.35 | 226.47 | 321.44 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 347.85 | 230.92 | 327.64 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=15.07 |
| Stop hit — per-position SL triggered | 2023-12-07 00:00:00 | 325.25 | 236.15 | 331.51 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 335.90 | 241.05 | 327.03 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=13.68 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 315.37 | 243.47 | 325.80 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 330.90 | 250.88 | 323.03 | Stage2 pullback-breakout RSI=55 vol=1.5x ATR=12.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 00:00:00 | 355.83 | 252.77 | 327.33 | T1 booked 50% @ 355.83 |
| Target hit | 2024-02-08 00:00:00 | 378.70 | 278.92 | 382.93 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 00:00:00 | 397.35 | 312.67 | 374.60 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=16.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 00:00:00 | 430.60 | 315.24 | 381.35 | T1 booked 50% @ 430.60 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 397.35 | 319.84 | 392.37 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-30 00:00:00 | 316.95 | 2023-11-12 00:00:00 | 320.90 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest1 | 2023-11-17 00:00:00 | 329.35 | 2023-11-21 00:00:00 | 354.77 | PARTIAL | 0.50 | 7.72% |
| BUY | retest1 | 2023-11-17 00:00:00 | 329.35 | 2023-11-23 00:00:00 | 329.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 00:00:00 | 347.85 | 2023-12-07 00:00:00 | 325.25 | STOP_HIT | 1.00 | -6.50% |
| BUY | retest1 | 2023-12-15 00:00:00 | 335.90 | 2023-12-20 00:00:00 | 315.37 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest1 | 2024-01-04 00:00:00 | 330.90 | 2024-01-08 00:00:00 | 355.83 | PARTIAL | 0.50 | 7.53% |
| BUY | retest1 | 2024-01-04 00:00:00 | 330.90 | 2024-02-08 00:00:00 | 378.70 | TARGET_HIT | 0.50 | 14.45% |
| BUY | retest1 | 2024-04-22 00:00:00 | 397.35 | 2024-04-25 00:00:00 | 430.60 | PARTIAL | 0.50 | 8.37% |
| BUY | retest1 | 2024-04-22 00:00:00 | 397.35 | 2024-05-03 00:00:00 | 397.35 | STOP_HIT | 0.50 | 0.00% |
