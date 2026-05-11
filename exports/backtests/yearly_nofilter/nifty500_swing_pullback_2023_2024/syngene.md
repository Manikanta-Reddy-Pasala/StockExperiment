# Syngene International Ltd. (SYNGENE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 458.10
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 0.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 6 | 1 | 0.02% | 0.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 6 | 1 | 0.02% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 6 | 1 | 0.02% | 0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 00:00:00 | 759.60 | 633.11 | 732.11 | Stage2 pullback-breakout RSI=69 vol=3.3x ATR=15.86 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 767.40 | 645.37 | 751.83 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-07-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 00:00:00 | 807.05 | 658.10 | 769.47 | Stage2 pullback-breakout RSI=66 vol=6.4x ATR=19.25 |
| Stop hit — per-position SL triggered | 2023-08-10 00:00:00 | 815.60 | 672.72 | 796.69 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 807.70 | 691.34 | 790.81 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=19.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 00:00:00 | 846.29 | 694.31 | 799.99 | T1 booked 50% @ 846.29 |
| Stop hit — per-position SL triggered | 2023-09-18 00:00:00 | 807.70 | 703.53 | 814.56 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-11-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 00:00:00 | 745.55 | 716.04 | 725.07 | Stage2 pullback-breakout RSI=59 vol=2.8x ATR=16.54 |
| Stop hit — per-position SL triggered | 2023-12-06 00:00:00 | 720.74 | 717.82 | 732.91 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 742.50 | 716.16 | 716.80 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=16.61 |
| Stop hit — per-position SL triggered | 2024-01-16 00:00:00 | 717.59 | 716.50 | 719.51 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 750.50 | 714.87 | 708.59 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=20.58 |
| Stop hit — per-position SL triggered | 2024-02-14 00:00:00 | 749.95 | 717.96 | 732.93 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 00:00:00 | 759.60 | 2023-07-13 00:00:00 | 767.40 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest1 | 2023-07-27 00:00:00 | 807.05 | 2023-08-10 00:00:00 | 815.60 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest1 | 2023-09-05 00:00:00 | 807.70 | 2023-09-07 00:00:00 | 846.29 | PARTIAL | 0.50 | 4.78% |
| BUY | retest1 | 2023-09-05 00:00:00 | 807.70 | 2023-09-18 00:00:00 | 807.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-24 00:00:00 | 745.55 | 2023-12-06 00:00:00 | 720.74 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest1 | 2024-01-11 00:00:00 | 742.50 | 2024-01-16 00:00:00 | 717.59 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest1 | 2024-01-31 00:00:00 | 750.50 | 2024-02-14 00:00:00 | 749.95 | STOP_HIT | 1.00 | -0.07% |
