# Five-Star Business Finance Ltd. (FIVESTAR)

## Backtest Summary

- **Window:** 2022-11-21 00:00:00 → 2026-05-11 00:00:00 (860 bars)
- **Last close:** 459.00
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 3.17% / 1.12%
- **Sum % (uncompounded):** 15.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 0 | 3 | 2 | 3.17% | 15.8% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 3 | 2 | 3.17% | 15.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 3 | 2 | 3.17% | 15.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 747.40 | 635.27 | 717.71 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=22.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 00:00:00 | 792.59 | 641.47 | 735.98 | T1 booked 50% @ 792.59 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 747.40 | 643.79 | 740.05 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 744.85 | 686.34 | 727.73 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=19.53 |
| Stop hit — per-position SL triggered | 2024-01-25 00:00:00 | 749.85 | 691.97 | 739.80 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-03-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 00:00:00 | 715.75 | 697.96 | 680.11 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=28.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 00:00:00 | 772.99 | 701.50 | 717.43 | T1 booked 50% @ 772.99 |
| Stop hit — per-position SL triggered | 2024-04-10 00:00:00 | 723.80 | 702.10 | 720.02 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 741.05 | 702.08 | 708.96 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=27.94 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-11 00:00:00 | 747.40 | 2023-10-18 00:00:00 | 792.59 | PARTIAL | 0.50 | 6.05% |
| BUY | retest1 | 2023-10-11 00:00:00 | 747.40 | 2023-10-20 00:00:00 | 747.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-11 00:00:00 | 744.85 | 2024-01-25 00:00:00 | 749.85 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest1 | 2024-03-22 00:00:00 | 715.75 | 2024-04-08 00:00:00 | 772.99 | PARTIAL | 0.50 | 8.00% |
| BUY | retest1 | 2024-03-22 00:00:00 | 715.75 | 2024-04-10 00:00:00 | 723.80 | STOP_HIT | 0.50 | 1.12% |
