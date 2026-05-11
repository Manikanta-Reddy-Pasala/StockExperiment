# Allied Blenders and Distillers Ltd. (ABDL)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 591.30
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 1.20% / 0.00%
- **Sum % (uncompounded):** 10.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.20% | 10.8% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.20% | 10.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.20% | 10.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 05:30:00 | 493.90 | 384.80 | 467.33 | Stage2 pullback-breakout RSI=65 vol=3.3x ATR=17.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 05:30:00 | 528.73 | 387.33 | 475.57 | T1 booked 50% @ 528.73 |
| Stop hit — per-position SL triggered | 2025-08-06 05:30:00 | 493.90 | 391.01 | 484.67 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2025-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 05:30:00 | 527.30 | 399.33 | 493.36 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=19.22 |
| Stop hit — per-position SL triggered | 2025-08-28 05:30:00 | 498.47 | 406.08 | 502.70 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2025-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 05:30:00 | 529.15 | 415.71 | 506.32 | Stage2 pullback-breakout RSI=62 vol=2.7x ATR=16.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 05:30:00 | 562.60 | 418.47 | 515.24 | T1 booked 50% @ 562.60 |
| Stop hit — per-position SL triggered | 2025-09-24 05:30:00 | 529.15 | 426.91 | 528.84 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2025-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 05:30:00 | 553.00 | 432.54 | 527.17 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=18.17 |
| Stop hit — per-position SL triggered | 2025-10-13 05:30:00 | 525.74 | 439.11 | 534.85 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2025-10-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 05:30:00 | 580.90 | 444.47 | 541.30 | Stage2 pullback-breakout RSI=67 vol=3.7x ATR=18.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 05:30:00 | 618.13 | 447.84 | 554.69 | T1 booked 50% @ 618.13 |
| Target hit | 2025-12-01 05:30:00 | 622.95 | 493.18 | 641.16 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-02-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 05:30:00 | 538.40 | 511.73 | 506.21 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=20.83 |
| Stop hit — per-position SL triggered | 2026-02-19 05:30:00 | 507.16 | 512.21 | 511.90 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-30 05:30:00 | 493.90 | 2025-08-01 05:30:00 | 528.73 | PARTIAL | 0.50 | 7.05% |
| BUY | retest1 | 2025-07-30 05:30:00 | 493.90 | 2025-08-06 05:30:00 | 493.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 05:30:00 | 527.30 | 2025-08-28 05:30:00 | 498.47 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest1 | 2025-09-11 05:30:00 | 529.15 | 2025-09-15 05:30:00 | 562.60 | PARTIAL | 0.50 | 6.32% |
| BUY | retest1 | 2025-09-11 05:30:00 | 529.15 | 2025-09-24 05:30:00 | 529.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-03 05:30:00 | 553.00 | 2025-10-13 05:30:00 | 525.74 | STOP_HIT | 1.00 | -4.93% |
| BUY | retest1 | 2025-10-20 05:30:00 | 580.90 | 2025-10-23 05:30:00 | 618.13 | PARTIAL | 0.50 | 6.41% |
| BUY | retest1 | 2025-10-20 05:30:00 | 580.90 | 2025-12-01 05:30:00 | 622.95 | TARGET_HIT | 0.50 | 7.24% |
| BUY | retest1 | 2026-02-12 05:30:00 | 538.40 | 2026-02-19 05:30:00 | 507.16 | STOP_HIT | 1.00 | -5.80% |
