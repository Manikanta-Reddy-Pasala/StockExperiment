# Gujarat Mineral Development Corporation Ltd. (GMDCLTD)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 684.70
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 4
- **Avg / median % per leg:** 12.27% / 5.24%
- **Sum % (uncompounded):** 122.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 1 | 5 | 4 | 12.27% | 122.7% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 1 | 5 | 4 | 12.27% | 122.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 6 | 60.0% | 1 | 5 | 4 | 12.27% | 122.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 05:30:00 | 174.35 | 153.76 | 167.38 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 05:30:00 | 183.48 | 155.00 | 172.17 | T1 booked 50% @ 183.48 |
| Stop hit — per-position SL triggered | 2023-07-28 05:30:00 | 178.45 | 156.27 | 175.50 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 05:30:00 | 177.25 | 157.77 | 173.23 | Stage2 pullback-breakout RSI=57 vol=1.5x ATR=5.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 05:30:00 | 188.37 | 159.14 | 177.34 | T1 booked 50% @ 188.37 |
| Target hit | 2023-10-23 05:30:00 | 352.80 | 210.97 | 357.03 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 05:30:00 | 432.75 | 279.76 | 408.76 | Stage2 pullback-breakout RSI=62 vol=3.0x ATR=20.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 05:30:00 | 473.75 | 289.55 | 427.44 | T1 booked 50% @ 473.75 |
| Stop hit — per-position SL triggered | 2024-01-23 05:30:00 | 432.75 | 306.25 | 450.35 | SL hit (bars_held=16) |

### Cycle 4 — BUY (started 2024-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 05:30:00 | 489.45 | 318.32 | 458.37 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=21.75 |
| Stop hit — per-position SL triggered | 2024-02-09 05:30:00 | 456.82 | 324.33 | 462.05 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 05:30:00 | 390.20 | 340.69 | 369.27 | Stage2 pullback-breakout RSI=55 vol=2.2x ATR=20.12 |
| Stop hit — per-position SL triggered | 2024-04-18 05:30:00 | 383.90 | 345.61 | 383.78 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 05:30:00 | 416.40 | 347.72 | 389.31 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=18.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 05:30:00 | 452.43 | 352.47 | 407.69 | T1 booked 50% @ 452.43 |
| Stop hit — per-position SL triggered | 2024-05-07 05:30:00 | 416.40 | 353.77 | 409.47 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 05:30:00 | 174.35 | 2023-07-21 05:30:00 | 183.48 | PARTIAL | 0.50 | 5.24% |
| BUY | retest1 | 2023-07-14 05:30:00 | 174.35 | 2023-07-28 05:30:00 | 178.45 | STOP_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2023-08-11 05:30:00 | 177.25 | 2023-08-22 05:30:00 | 188.37 | PARTIAL | 0.50 | 6.28% |
| BUY | retest1 | 2023-08-11 05:30:00 | 177.25 | 2023-10-23 05:30:00 | 352.80 | TARGET_HIT | 0.50 | 99.04% |
| BUY | retest1 | 2024-01-01 05:30:00 | 432.75 | 2024-01-09 05:30:00 | 473.75 | PARTIAL | 0.50 | 9.47% |
| BUY | retest1 | 2024-01-01 05:30:00 | 432.75 | 2024-01-23 05:30:00 | 432.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-05 05:30:00 | 489.45 | 2024-02-09 05:30:00 | 456.82 | STOP_HIT | 1.00 | -6.67% |
| BUY | retest1 | 2024-04-02 05:30:00 | 390.20 | 2024-04-18 05:30:00 | 383.90 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest1 | 2024-04-24 05:30:00 | 416.40 | 2024-05-03 05:30:00 | 452.43 | PARTIAL | 0.50 | 8.65% |
| BUY | retest1 | 2024-04-24 05:30:00 | 416.40 | 2024-05-07 05:30:00 | 416.40 | STOP_HIT | 0.50 | 0.00% |
