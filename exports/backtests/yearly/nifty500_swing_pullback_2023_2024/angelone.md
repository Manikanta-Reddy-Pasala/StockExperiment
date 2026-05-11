# Angel One Ltd. (ANGELONE)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 326.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 13.33% / 0.60%
- **Sum % (uncompounded):** 66.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | 13.33% | 66.7% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 13.33% | 66.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | 13.33% | 66.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 05:30:00 | 170.10 | 140.37 | 157.64 | Stage2 pullback-breakout RSI=62 vol=2.7x ATR=6.61 |
| Stop hit — per-position SL triggered | 2023-08-24 05:30:00 | 171.12 | 143.99 | 169.31 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 05:30:00 | 194.37 | 152.29 | 184.01 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=6.77 |
| Stop hit — per-position SL triggered | 2023-09-29 05:30:00 | 184.21 | 152.96 | 184.41 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-10-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 05:30:00 | 198.92 | 154.04 | 185.88 | Stage2 pullback-breakout RSI=64 vol=3.3x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 05:30:00 | 212.83 | 156.04 | 192.29 | T1 booked 50% @ 212.83 |
| Target hit | 2024-01-16 05:30:00 | 332.99 | 229.20 | 353.54 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 05:30:00 | 297.31 | 254.88 | 274.90 | Stage2 pullback-breakout RSI=58 vol=2.5x ATR=15.30 |
| Stop hit — per-position SL triggered | 2024-04-12 05:30:00 | 288.10 | 259.00 | 288.53 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-08 05:30:00 | 170.10 | 2023-08-24 05:30:00 | 171.12 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest1 | 2023-09-27 05:30:00 | 194.37 | 2023-09-29 05:30:00 | 184.21 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest1 | 2023-10-05 05:30:00 | 198.92 | 2023-10-11 05:30:00 | 212.83 | PARTIAL | 0.50 | 6.99% |
| BUY | retest1 | 2023-10-05 05:30:00 | 198.92 | 2024-01-16 05:30:00 | 332.99 | TARGET_HIT | 0.50 | 67.40% |
| BUY | retest1 | 2024-03-27 05:30:00 | 297.31 | 2024-04-12 05:30:00 | 288.10 | STOP_HIT | 1.00 | -3.10% |
