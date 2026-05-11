# Choice International Ltd. (CHOICEIN)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 688.85
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
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -1.29% / -2.95%
- **Sum % (uncompounded):** -9.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -1.29% | -9.0% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -1.29% | -9.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 6 | 1 | -1.29% | -9.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 05:30:00 | 734.40 | 571.89 | 701.01 | Stage2 pullback-breakout RSI=70 vol=2.5x ATR=16.48 |
| Stop hit — per-position SL triggered | 2025-07-07 05:30:00 | 709.68 | 573.26 | 701.80 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2025-07-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 05:30:00 | 747.10 | 582.38 | 705.93 | Stage2 pullback-breakout RSI=66 vol=2.5x ATR=19.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 05:30:00 | 786.37 | 589.29 | 723.45 | T1 booked 50% @ 786.37 |
| Stop hit — per-position SL triggered | 2025-07-23 05:30:00 | 747.10 | 591.01 | 727.05 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-10-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 05:30:00 | 821.50 | 675.62 | 800.04 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=18.95 |
| Stop hit — per-position SL triggered | 2025-10-28 05:30:00 | 816.85 | 688.99 | 809.73 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-10-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 05:30:00 | 836.60 | 693.00 | 813.69 | Stage2 pullback-breakout RSI=62 vol=3.0x ATR=16.44 |
| Stop hit — per-position SL triggered | 2025-11-04 05:30:00 | 811.94 | 695.51 | 814.70 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2025-11-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 05:30:00 | 811.05 | 708.84 | 796.20 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=16.59 |
| Stop hit — per-position SL triggered | 2025-12-02 05:30:00 | 786.17 | 712.40 | 797.00 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2026-02-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 05:30:00 | 795.40 | 742.90 | 774.98 | Stage2 pullback-breakout RSI=55 vol=2.1x ATR=23.09 |
| Stop hit — per-position SL triggered | 2026-02-24 05:30:00 | 760.77 | 744.45 | 776.87 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-04 05:30:00 | 734.40 | 2025-07-07 05:30:00 | 709.68 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest1 | 2025-07-16 05:30:00 | 747.10 | 2025-07-22 05:30:00 | 786.37 | PARTIAL | 0.50 | 5.26% |
| BUY | retest1 | 2025-07-16 05:30:00 | 747.10 | 2025-07-23 05:30:00 | 747.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-13 05:30:00 | 821.50 | 2025-10-28 05:30:00 | 816.85 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-10-31 05:30:00 | 836.60 | 2025-11-04 05:30:00 | 811.94 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest1 | 2025-11-26 05:30:00 | 811.05 | 2025-12-02 05:30:00 | 786.17 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest1 | 2026-02-18 05:30:00 | 795.40 | 2026-02-24 05:30:00 | 760.77 | STOP_HIT | 1.00 | -4.35% |
