# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 452.15
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
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** -0.28% / 0.00%
- **Sum % (uncompounded):** -1.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.28% | -2.0% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.28% | -2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.28% | -2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 00:00:00 | 359.80 | 297.74 | 338.70 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=11.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 00:00:00 | 383.77 | 299.96 | 347.69 | T1 booked 50% @ 383.77 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 359.80 | 311.27 | 374.28 | SL hit (bars_held=18) |

### Cycle 2 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 377.65 | 317.74 | 366.61 | Stage2 pullback-breakout RSI=56 vol=2.8x ATR=14.61 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 355.74 | 318.62 | 365.82 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-03-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 00:00:00 | 372.75 | 326.09 | 360.22 | Stage2 pullback-breakout RSI=58 vol=4.6x ATR=11.44 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 355.59 | 326.99 | 358.94 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 374.10 | 329.61 | 353.13 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=11.22 |
| Stop hit — per-position SL triggered | 2024-04-19 00:00:00 | 357.27 | 333.73 | 365.20 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 402.20 | 335.25 | 370.59 | Stage2 pullback-breakout RSI=68 vol=7.0x ATR=12.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 00:00:00 | 427.44 | 337.74 | 383.42 | T1 booked 50% @ 427.44 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 402.20 | 340.03 | 391.49 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-22 00:00:00 | 359.80 | 2023-12-28 00:00:00 | 383.77 | PARTIAL | 0.50 | 6.66% |
| BUY | retest1 | 2023-12-22 00:00:00 | 359.80 | 2024-01-18 00:00:00 | 359.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-07 00:00:00 | 377.65 | 2024-02-09 00:00:00 | 355.74 | STOP_HIT | 1.00 | -5.80% |
| BUY | retest1 | 2024-03-07 00:00:00 | 372.75 | 2024-03-13 00:00:00 | 355.59 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest1 | 2024-04-03 00:00:00 | 374.10 | 2024-04-19 00:00:00 | 357.27 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest1 | 2024-04-24 00:00:00 | 402.20 | 2024-04-29 00:00:00 | 427.44 | PARTIAL | 0.50 | 6.27% |
| BUY | retest1 | 2024-04-24 00:00:00 | 402.20 | 2024-05-03 00:00:00 | 402.20 | STOP_HIT | 0.50 | 0.00% |
