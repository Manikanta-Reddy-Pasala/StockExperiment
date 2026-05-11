# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 3323.80
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
- **Avg / median % per leg:** -1.05% / -4.17%
- **Sum % (uncompounded):** -7.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 5 | 1 | -1.05% | -7.4% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 5 | 1 | -1.05% | -7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 5 | 1 | -1.05% | -7.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 05:30:00 | 3818.10 | 3195.04 | 3663.67 | Stage2 pullback-breakout RSI=62 vol=4.8x ATR=115.15 |
| Stop hit — per-position SL triggered | 2025-07-21 05:30:00 | 3645.38 | 3248.36 | 3748.05 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2025-09-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 05:30:00 | 3376.00 | 3244.36 | 3229.83 | Stage2 pullback-breakout RSI=57 vol=3.5x ATR=108.92 |
| Stop hit — per-position SL triggered | 2025-09-16 05:30:00 | 3445.90 | 3254.73 | 3309.71 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 05:30:00 | 3699.00 | 3297.76 | 3487.45 | Stage2 pullback-breakout RSI=67 vol=2.5x ATR=97.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 05:30:00 | 3893.87 | 3311.06 | 3576.79 | T1 booked 50% @ 3893.87 |
| Target hit | 2025-11-18 05:30:00 | 3885.60 | 3440.71 | 3963.49 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-12-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 05:30:00 | 3982.70 | 3486.45 | 3911.24 | Stage2 pullback-breakout RSI=57 vol=2.7x ATR=110.59 |
| Stop hit — per-position SL triggered | 2025-12-08 05:30:00 | 3816.81 | 3497.96 | 3901.66 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 3931.70 | 3523.45 | 3839.14 | Stage2 pullback-breakout RSI=56 vol=4.1x ATR=115.31 |
| Stop hit — per-position SL triggered | 2025-12-30 05:30:00 | 3758.74 | 3542.40 | 3838.63 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2026-03-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 05:30:00 | 3806.50 | 3620.10 | 3624.16 | Stage2 pullback-breakout RSI=56 vol=4.1x ATR=168.86 |
| Stop hit — per-position SL triggered | 2026-03-13 05:30:00 | 3553.20 | 3618.61 | 3609.53 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-08 05:30:00 | 3818.10 | 2025-07-21 05:30:00 | 3645.38 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest1 | 2025-09-02 05:30:00 | 3376.00 | 2025-09-16 05:30:00 | 3445.90 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest1 | 2025-10-16 05:30:00 | 3699.00 | 2025-10-20 05:30:00 | 3893.87 | PARTIAL | 0.50 | 5.27% |
| BUY | retest1 | 2025-10-16 05:30:00 | 3699.00 | 2025-11-18 05:30:00 | 3885.60 | TARGET_HIT | 0.50 | 5.04% |
| BUY | retest1 | 2025-12-03 05:30:00 | 3982.70 | 2025-12-08 05:30:00 | 3816.81 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest1 | 2025-12-19 05:30:00 | 3931.70 | 2025-12-30 05:30:00 | 3758.74 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest1 | 2026-03-12 05:30:00 | 3806.50 | 2026-03-13 05:30:00 | 3553.20 | STOP_HIT | 1.00 | -6.65% |
