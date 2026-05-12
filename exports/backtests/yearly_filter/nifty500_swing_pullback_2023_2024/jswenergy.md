# JSW Energy Ltd. (JSWENERGY)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 554.55
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 8.90% / 6.74%
- **Sum % (uncompounded):** 53.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 2 | 2 | 2 | 8.90% | 53.4% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 2 | 2 | 2 | 8.90% | 53.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 2 | 2 | 2 | 8.90% | 53.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 00:00:00 | 314.25 | 280.92 | 293.20 | Stage2 pullback-breakout RSI=65 vol=4.4x ATR=10.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 00:00:00 | 335.43 | 281.47 | 297.33 | T1 booked 50% @ 335.43 |
| Target hit | 2023-10-10 00:00:00 | 408.55 | 317.77 | 409.76 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 399.90 | 333.73 | 391.16 | Stage2 pullback-breakout RSI=55 vol=2.1x ATR=15.87 |
| Stop hit — per-position SL triggered | 2023-11-30 00:00:00 | 408.35 | 340.85 | 402.91 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 00:00:00 | 454.85 | 361.27 | 422.28 | Stage2 pullback-breakout RSI=65 vol=4.4x ATR=16.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 00:00:00 | 488.50 | 370.22 | 453.48 | T1 booked 50% @ 488.50 |
| Target hit | 2024-02-14 00:00:00 | 479.70 | 389.46 | 485.57 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 506.85 | 396.30 | 488.59 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=18.94 |
| Stop hit — per-position SL triggered | 2024-03-07 00:00:00 | 515.25 | 407.34 | 503.40 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-09 00:00:00 | 314.25 | 2023-08-10 00:00:00 | 335.43 | PARTIAL | 0.50 | 6.74% |
| BUY | retest1 | 2023-08-09 00:00:00 | 314.25 | 2023-10-10 00:00:00 | 408.55 | TARGET_HIT | 0.50 | 30.01% |
| BUY | retest1 | 2023-11-15 00:00:00 | 399.90 | 2023-11-30 00:00:00 | 408.35 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest1 | 2024-01-09 00:00:00 | 454.85 | 2024-01-19 00:00:00 | 488.50 | PARTIAL | 0.50 | 7.40% |
| BUY | retest1 | 2024-01-09 00:00:00 | 454.85 | 2024-02-14 00:00:00 | 479.70 | TARGET_HIT | 0.50 | 5.46% |
| BUY | retest1 | 2024-02-23 00:00:00 | 506.85 | 2024-03-07 00:00:00 | 515.25 | STOP_HIT | 1.00 | 1.66% |
