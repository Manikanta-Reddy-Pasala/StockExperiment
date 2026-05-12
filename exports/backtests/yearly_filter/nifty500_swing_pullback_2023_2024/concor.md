# Container Corporation of India Ltd. (CONCOR)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 532.65
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
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 4
- **Avg / median % per leg:** 3.89% / 4.83%
- **Sum % (uncompounded):** 38.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 2 | 4 | 4 | 3.89% | 38.9% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 4 | 4 | 3.89% | 38.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 6 | 60.0% | 2 | 4 | 4 | 3.89% | 38.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 00:00:00 | 568.52 | 531.42 | 549.08 | Stage2 pullback-breakout RSI=66 vol=1.5x ATR=12.97 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 549.06 | 532.13 | 550.58 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-09-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 00:00:00 | 567.12 | 533.36 | 545.60 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=13.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 00:00:00 | 594.51 | 533.89 | 549.53 | T1 booked 50% @ 594.51 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 567.12 | 534.15 | 550.49 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 571.00 | 544.10 | 560.03 | Stage2 pullback-breakout RSI=56 vol=4.2x ATR=13.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 598.61 | 547.83 | 578.27 | T1 booked 50% @ 598.61 |
| Target hit | 2024-01-17 00:00:00 | 680.12 | 591.10 | 695.94 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 00:00:00 | 736.96 | 600.39 | 693.87 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=26.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 00:00:00 | 790.88 | 617.96 | 740.68 | T1 booked 50% @ 790.88 |
| Target hit | 2024-03-07 00:00:00 | 771.76 | 639.86 | 772.38 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 776.56 | 654.10 | 730.35 | Stage2 pullback-breakout RSI=64 vol=2.6x ATR=24.56 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 739.72 | 656.00 | 733.90 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 803.84 | 663.21 | 749.96 | Stage2 pullback-breakout RSI=66 vol=2.1x ATR=25.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 853.90 | 665.11 | 759.91 | T1 booked 50% @ 853.90 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 803.84 | 674.68 | 790.69 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-08 00:00:00 | 568.52 | 2023-08-11 00:00:00 | 549.06 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest1 | 2023-09-08 00:00:00 | 567.12 | 2023-09-11 00:00:00 | 594.51 | PARTIAL | 0.50 | 4.83% |
| BUY | retest1 | 2023-09-08 00:00:00 | 567.12 | 2023-09-12 00:00:00 | 567.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-03 00:00:00 | 571.00 | 2023-11-15 00:00:00 | 598.61 | PARTIAL | 0.50 | 4.84% |
| BUY | retest1 | 2023-11-03 00:00:00 | 571.00 | 2024-01-17 00:00:00 | 680.12 | TARGET_HIT | 0.50 | 19.11% |
| BUY | retest1 | 2024-02-01 00:00:00 | 736.96 | 2024-02-19 00:00:00 | 790.88 | PARTIAL | 0.50 | 7.32% |
| BUY | retest1 | 2024-02-01 00:00:00 | 736.96 | 2024-03-07 00:00:00 | 771.76 | TARGET_HIT | 0.50 | 4.72% |
| BUY | retest1 | 2024-04-10 00:00:00 | 776.56 | 2024-04-15 00:00:00 | 739.72 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest1 | 2024-04-25 00:00:00 | 803.84 | 2024-04-26 00:00:00 | 853.90 | PARTIAL | 0.50 | 6.23% |
| BUY | retest1 | 2024-04-25 00:00:00 | 803.84 | 2024-05-07 00:00:00 | 803.84 | STOP_HIT | 0.50 | 0.00% |
