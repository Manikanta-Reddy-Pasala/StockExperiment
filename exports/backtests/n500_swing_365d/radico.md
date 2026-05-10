# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 3477.20
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
- **Avg / median % per leg:** -1.11% / -3.15%
- **Sum % (uncompounded):** -7.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 6 | 1 | -1.11% | -7.7% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 6 | 1 | -1.11% | -7.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 6 | 1 | -1.11% | -7.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 05:30:00 | 2698.40 | 2373.21 | 2625.68 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=70.77 |
| Stop hit — per-position SL triggered | 2025-07-24 05:30:00 | 2726.30 | 2407.97 | 2697.44 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 05:30:00 | 2838.80 | 2426.51 | 2713.60 | Stage2 pullback-breakout RSI=69 vol=7.2x ATR=73.02 |
| Stop hit — per-position SL triggered | 2025-08-06 05:30:00 | 2729.27 | 2438.76 | 2746.96 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-09-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 05:30:00 | 2980.60 | 2534.13 | 2849.02 | Stage2 pullback-breakout RSI=65 vol=3.0x ATR=82.27 |
| Stop hit — per-position SL triggered | 2025-09-29 05:30:00 | 2886.60 | 2576.01 | 2920.92 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-10-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 05:30:00 | 3108.50 | 2624.41 | 2966.84 | Stage2 pullback-breakout RSI=67 vol=3.7x ATR=71.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 05:30:00 | 3251.97 | 2630.68 | 2994.25 | T1 booked 50% @ 3251.97 |
| Stop hit — per-position SL triggered | 2025-11-03 05:30:00 | 3193.60 | 2679.98 | 3112.47 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-11-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 05:30:00 | 3368.00 | 2745.99 | 3223.66 | Stage2 pullback-breakout RSI=64 vol=9.6x ATR=100.93 |
| Stop hit — per-position SL triggered | 2025-11-27 05:30:00 | 3216.60 | 2770.73 | 3233.86 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 3376.00 | 2860.48 | 3243.01 | Stage2 pullback-breakout RSI=61 vol=2.5x ATR=104.15 |
| Stop hit — per-position SL triggered | 2026-01-02 05:30:00 | 3219.77 | 2870.95 | 3234.10 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-10 05:30:00 | 2698.40 | 2025-07-24 05:30:00 | 2726.30 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest1 | 2025-08-01 05:30:00 | 2838.80 | 2025-08-06 05:30:00 | 2729.27 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest1 | 2025-09-15 05:30:00 | 2980.60 | 2025-09-29 05:30:00 | 2886.60 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest1 | 2025-10-17 05:30:00 | 3108.50 | 2025-10-20 05:30:00 | 3251.97 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2025-10-17 05:30:00 | 3108.50 | 2025-11-03 05:30:00 | 3193.60 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest1 | 2025-11-20 05:30:00 | 3368.00 | 2025-11-27 05:30:00 | 3216.60 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest1 | 2025-12-30 05:30:00 | 3376.00 | 2026-01-02 05:30:00 | 3219.77 | STOP_HIT | 1.00 | -4.63% |
