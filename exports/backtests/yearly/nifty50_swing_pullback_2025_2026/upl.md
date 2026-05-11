# UPL (UPL)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 646.00
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 2.83% / 4.07%
- **Sum % (uncompounded):** 19.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 1 | 3 | 3 | 2.83% | 19.8% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 1 | 3 | 3 | 2.83% | 19.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 1 | 3 | 3 | 2.83% | 19.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 05:30:00 | 671.05 | 611.46 | 645.14 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=13.70 |
| Stop hit — per-position SL triggered | 2025-07-11 05:30:00 | 650.49 | 616.05 | 658.47 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2025-07-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 05:30:00 | 692.65 | 618.34 | 664.11 | Stage2 pullback-breakout RSI=66 vol=2.5x ATR=14.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 05:30:00 | 722.15 | 620.99 | 675.76 | T1 booked 50% @ 722.15 |
| Stop hit — per-position SL triggered | 2025-07-31 05:30:00 | 703.80 | 627.85 | 698.73 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 702.60 | 655.08 | 680.11 | Stage2 pullback-breakout RSI=62 vol=3.0x ATR=14.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 05:30:00 | 731.22 | 657.75 | 694.40 | T1 booked 50% @ 731.22 |
| Target hit | 2025-12-03 05:30:00 | 743.00 | 675.63 | 745.16 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-12-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 05:30:00 | 776.55 | 684.81 | 750.42 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=16.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 05:30:00 | 809.39 | 692.64 | 771.69 | T1 booked 50% @ 809.39 |
| Stop hit — per-position SL triggered | 2026-01-06 05:30:00 | 799.30 | 694.81 | 777.20 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-01 05:30:00 | 671.05 | 2025-07-11 05:30:00 | 650.49 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest1 | 2025-07-17 05:30:00 | 692.65 | 2025-07-22 05:30:00 | 722.15 | PARTIAL | 0.50 | 4.26% |
| BUY | retest1 | 2025-07-17 05:30:00 | 692.65 | 2025-07-31 05:30:00 | 703.80 | STOP_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2025-10-28 05:30:00 | 702.60 | 2025-11-03 05:30:00 | 731.22 | PARTIAL | 0.50 | 4.07% |
| BUY | retest1 | 2025-10-28 05:30:00 | 702.60 | 2025-12-03 05:30:00 | 743.00 | TARGET_HIT | 0.50 | 5.75% |
| BUY | retest1 | 2025-12-22 05:30:00 | 776.55 | 2026-01-02 05:30:00 | 809.39 | PARTIAL | 0.50 | 4.23% |
| BUY | retest1 | 2025-12-22 05:30:00 | 776.55 | 2026-01-06 05:30:00 | 799.30 | STOP_HIT | 0.50 | 2.93% |
