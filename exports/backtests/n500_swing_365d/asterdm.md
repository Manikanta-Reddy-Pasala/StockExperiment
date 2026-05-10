# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 742.40
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
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 1.85% / 5.64%
- **Sum % (uncompounded):** 14.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.85% | 14.8% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.85% | 14.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.85% | 14.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 05:30:00 | 604.90 | 512.46 | 594.53 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=18.74 |
| Stop hit — per-position SL triggered | 2025-08-08 05:30:00 | 576.79 | 517.11 | 593.31 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2025-09-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 05:30:00 | 632.05 | 529.71 | 604.84 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=19.10 |
| Stop hit — per-position SL triggered | 2025-09-16 05:30:00 | 619.90 | 539.49 | 621.69 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-09-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 05:30:00 | 650.55 | 542.06 | 623.19 | Stage2 pullback-breakout RSI=63 vol=3.1x ATR=17.99 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 623.57 | 546.69 | 628.34 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2025-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 05:30:00 | 672.55 | 551.24 | 633.82 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=18.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 05:30:00 | 710.45 | 555.23 | 647.65 | T1 booked 50% @ 710.45 |
| Stop hit — per-position SL triggered | 2025-10-31 05:30:00 | 672.55 | 575.53 | 688.51 | SL hit (bars_held=18) |

### Cycle 5 — BUY (started 2026-02-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 05:30:00 | 600.40 | 592.86 | 571.28 | Stage2 pullback-breakout RSI=58 vol=2.8x ATR=24.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 05:30:00 | 648.48 | 595.71 | 605.47 | T1 booked 50% @ 648.48 |
| Target hit | 2026-03-19 05:30:00 | 636.20 | 605.11 | 649.06 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-04-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 05:30:00 | 705.25 | 617.85 | 674.71 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=20.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 05:30:00 | 746.79 | 622.55 | 690.63 | T1 booked 50% @ 746.79 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-31 05:30:00 | 604.90 | 2025-08-08 05:30:00 | 576.79 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest1 | 2025-09-02 05:30:00 | 632.05 | 2025-09-16 05:30:00 | 619.90 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest1 | 2025-09-19 05:30:00 | 650.55 | 2025-09-26 05:30:00 | 623.57 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest1 | 2025-10-06 05:30:00 | 672.55 | 2025-10-09 05:30:00 | 710.45 | PARTIAL | 0.50 | 5.64% |
| BUY | retest1 | 2025-10-06 05:30:00 | 672.55 | 2025-10-31 05:30:00 | 672.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 05:30:00 | 600.40 | 2026-02-24 05:30:00 | 648.48 | PARTIAL | 0.50 | 8.01% |
| BUY | retest1 | 2026-02-11 05:30:00 | 600.40 | 2026-03-19 05:30:00 | 636.20 | TARGET_HIT | 0.50 | 5.96% |
| BUY | retest1 | 2026-04-24 05:30:00 | 705.25 | 2026-05-04 05:30:00 | 746.79 | PARTIAL | 0.50 | 5.89% |
