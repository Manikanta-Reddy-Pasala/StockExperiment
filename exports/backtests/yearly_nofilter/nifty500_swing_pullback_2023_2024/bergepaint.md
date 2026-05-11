# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 502.80
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -1.22% / -2.96%
- **Sum % (uncompounded):** -6.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.22% | -6.1% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.22% | -6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.22% | -6.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 00:00:00 | 585.33 | 525.26 | 565.63 | Stage2 pullback-breakout RSI=68 vol=3.1x ATR=11.53 |
| Stop hit — per-position SL triggered | 2023-08-10 00:00:00 | 568.03 | 529.36 | 575.59 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-09-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 00:00:00 | 607.50 | 543.03 | 594.87 | Stage2 pullback-breakout RSI=64 vol=1.7x ATR=12.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 00:00:00 | 632.15 | 545.11 | 599.87 | T1 booked 50% @ 632.15 |
| Stop hit — per-position SL triggered | 2023-09-26 00:00:00 | 607.50 | 548.32 | 608.20 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 586.15 | 556.16 | 575.23 | Stage2 pullback-breakout RSI=58 vol=3.4x ATR=12.18 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 567.88 | 557.97 | 577.05 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-02-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 00:00:00 | 606.70 | 565.03 | 568.91 | Stage2 pullback-breakout RSI=69 vol=5.7x ATR=16.62 |
| Stop hit — per-position SL triggered | 2024-03-01 00:00:00 | 581.77 | 565.27 | 570.84 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-01 00:00:00 | 585.33 | 2023-08-10 00:00:00 | 568.03 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest1 | 2023-09-14 00:00:00 | 607.50 | 2023-09-20 00:00:00 | 632.15 | PARTIAL | 0.50 | 4.06% |
| BUY | retest1 | 2023-09-14 00:00:00 | 607.50 | 2023-09-26 00:00:00 | 607.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-01 00:00:00 | 586.15 | 2023-12-13 00:00:00 | 567.88 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest1 | 2024-02-29 00:00:00 | 606.70 | 2024-03-01 00:00:00 | 581.77 | STOP_HIT | 1.00 | -4.11% |
