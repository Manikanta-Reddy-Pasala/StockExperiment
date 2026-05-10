# Asahi India Glass Ltd. (ASAHIINDIA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 836.10
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
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** -0.69% / -0.06%
- **Sum % (uncompounded):** -4.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 5 | 1 | -0.69% | -4.8% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 5 | 1 | -0.69% | -4.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 5 | 1 | -0.69% | -4.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 05:30:00 | 806.30 | 690.93 | 738.29 | Stage2 pullback-breakout RSI=66 vol=8.6x ATR=31.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 05:30:00 | 870.04 | 707.27 | 808.66 | T1 booked 50% @ 870.04 |
| Target hit | 2025-08-01 05:30:00 | 814.80 | 720.94 | 827.66 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 867.90 | 731.57 | 831.65 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=27.16 |
| Stop hit — per-position SL triggered | 2025-09-02 05:30:00 | 843.65 | 744.27 | 851.15 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-09-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 05:30:00 | 875.70 | 748.67 | 845.38 | Stage2 pullback-breakout RSI=58 vol=3.6x ATR=29.56 |
| Stop hit — per-position SL triggered | 2025-09-29 05:30:00 | 875.20 | 767.14 | 881.86 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-10-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 05:30:00 | 913.70 | 780.50 | 887.31 | Stage2 pullback-breakout RSI=59 vol=4.1x ATR=30.28 |
| Stop hit — per-position SL triggered | 2025-10-31 05:30:00 | 922.90 | 795.49 | 919.15 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 1037.60 | 862.59 | 999.66 | Stage2 pullback-breakout RSI=58 vol=5.8x ATR=39.11 |
| Stop hit — per-position SL triggered | 2026-01-05 05:30:00 | 978.93 | 867.75 | 997.31 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 1001.80 | 881.89 | 967.45 | Stage2 pullback-breakout RSI=55 vol=3.0x ATR=41.94 |
| Stop hit — per-position SL triggered | 2026-02-06 05:30:00 | 938.89 | 887.67 | 973.02 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-01 05:30:00 | 806.30 | 2025-07-17 05:30:00 | 870.04 | PARTIAL | 0.50 | 7.91% |
| BUY | retest1 | 2025-07-01 05:30:00 | 806.30 | 2025-08-01 05:30:00 | 814.80 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2025-08-18 05:30:00 | 867.90 | 2025-09-02 05:30:00 | 843.65 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest1 | 2025-09-09 05:30:00 | 875.70 | 2025-09-29 05:30:00 | 875.20 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest1 | 2025-10-16 05:30:00 | 913.70 | 2025-10-31 05:30:00 | 922.90 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest1 | 2025-12-30 05:30:00 | 1037.60 | 2026-01-05 05:30:00 | 978.93 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest1 | 2026-01-30 05:30:00 | 1001.80 | 2026-02-06 05:30:00 | 938.89 | STOP_HIT | 1.00 | -6.28% |
