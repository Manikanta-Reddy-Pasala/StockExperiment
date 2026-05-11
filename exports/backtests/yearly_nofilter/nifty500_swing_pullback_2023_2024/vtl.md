# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 580.70
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 2.11% / 2.33%
- **Sum % (uncompounded):** 16.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 4 | 2 | 2.11% | 16.9% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 4 | 2 | 2.11% | 16.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 4 | 2 | 2.11% | 16.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 00:00:00 | 377.85 | 330.49 | 362.51 | Stage2 pullback-breakout RSI=64 vol=3.7x ATR=11.82 |
| Stop hit — per-position SL triggered | 2023-07-26 00:00:00 | 370.55 | 334.46 | 368.62 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 00:00:00 | 366.15 | 336.82 | 352.77 | Stage2 pullback-breakout RSI=57 vol=6.5x ATR=11.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 00:00:00 | 389.56 | 339.46 | 365.16 | T1 booked 50% @ 389.56 |
| Target hit | 2023-09-12 00:00:00 | 376.30 | 345.10 | 385.84 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 385.60 | 355.02 | 368.86 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=13.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 00:00:00 | 412.05 | 355.94 | 374.96 | T1 booked 50% @ 412.05 |
| Target hit | 2023-12-08 00:00:00 | 394.60 | 361.56 | 396.88 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 399.20 | 367.47 | 389.15 | Stage2 pullback-breakout RSI=57 vol=2.6x ATR=11.63 |
| Stop hit — per-position SL triggered | 2024-01-25 00:00:00 | 397.05 | 370.57 | 395.37 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 427.80 | 371.68 | 398.91 | Stage2 pullback-breakout RSI=67 vol=7.0x ATR=14.65 |
| Stop hit — per-position SL triggered | 2024-02-15 00:00:00 | 434.00 | 377.99 | 421.66 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 456.15 | 394.02 | 436.94 | Stage2 pullback-breakout RSI=63 vol=4.9x ATR=15.94 |
| Stop hit — per-position SL triggered | 2024-04-22 00:00:00 | 454.15 | 400.28 | 451.95 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 00:00:00 | 377.85 | 2023-07-26 00:00:00 | 370.55 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest1 | 2023-08-18 00:00:00 | 366.15 | 2023-08-29 00:00:00 | 389.56 | PARTIAL | 0.50 | 6.39% |
| BUY | retest1 | 2023-08-18 00:00:00 | 366.15 | 2023-09-12 00:00:00 | 376.30 | TARGET_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2023-11-20 00:00:00 | 385.60 | 2023-11-22 00:00:00 | 412.05 | PARTIAL | 0.50 | 6.86% |
| BUY | retest1 | 2023-11-20 00:00:00 | 385.60 | 2023-12-08 00:00:00 | 394.60 | TARGET_HIT | 0.50 | 2.33% |
| BUY | retest1 | 2024-01-11 00:00:00 | 399.20 | 2024-01-25 00:00:00 | 397.05 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-01-31 00:00:00 | 427.80 | 2024-02-15 00:00:00 | 434.00 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest1 | 2024-04-04 00:00:00 | 456.15 | 2024-04-22 00:00:00 | 454.15 | STOP_HIT | 1.00 | -0.44% |
