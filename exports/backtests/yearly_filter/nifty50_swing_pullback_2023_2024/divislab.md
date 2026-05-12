# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 6710.50
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 1.27% / 1.86%
- **Sum % (uncompounded):** 7.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.27% | 7.6% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.27% | 7.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.27% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 00:00:00 | 3747.95 | 3330.62 | 3547.09 | Stage2 pullback-breakout RSI=68 vol=3.1x ATR=85.33 |
| Stop hit — per-position SL triggered | 2023-07-10 00:00:00 | 3619.96 | 3340.99 | 3581.20 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-09-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-12 00:00:00 | 3787.80 | 3466.27 | 3684.84 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=72.22 |
| Stop hit — per-position SL triggered | 2023-09-27 00:00:00 | 3768.00 | 3494.66 | 3732.62 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 3672.95 | 3515.05 | 3534.98 | Stage2 pullback-breakout RSI=66 vol=3.2x ATR=77.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 3827.38 | 3535.31 | 3667.29 | T1 booked 50% @ 3827.38 |
| Stop hit — per-position SL triggered | 2023-12-05 00:00:00 | 3741.15 | 3537.35 | 3674.32 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-12-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 00:00:00 | 3863.50 | 3558.21 | 3695.61 | Stage2 pullback-breakout RSI=67 vol=2.9x ATR=86.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 00:00:00 | 4035.72 | 3576.61 | 3791.95 | T1 booked 50% @ 4035.72 |
| Stop hit — per-position SL triggered | 2024-01-10 00:00:00 | 3902.65 | 3600.34 | 3875.65 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-05 00:00:00 | 3747.95 | 2023-07-10 00:00:00 | 3619.96 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest1 | 2023-09-12 00:00:00 | 3787.80 | 2023-09-27 00:00:00 | 3768.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2023-11-20 00:00:00 | 3672.95 | 2023-12-04 00:00:00 | 3827.38 | PARTIAL | 0.50 | 4.20% |
| BUY | retest1 | 2023-11-20 00:00:00 | 3672.95 | 2023-12-05 00:00:00 | 3741.15 | STOP_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2023-12-26 00:00:00 | 3863.50 | 2024-01-02 00:00:00 | 4035.72 | PARTIAL | 0.50 | 4.46% |
| BUY | retest1 | 2023-12-26 00:00:00 | 3863.50 | 2024-01-10 00:00:00 | 3902.65 | STOP_HIT | 0.50 | 1.01% |
