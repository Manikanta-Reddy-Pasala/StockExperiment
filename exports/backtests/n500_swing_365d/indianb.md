# Indian Bank (INDIANB)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 865.55
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
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.52% / 2.28%
- **Sum % (uncompounded):** 3.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.52% | 3.1% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.52% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 4 | 2 | 0.52% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 05:30:00 | 652.15 | 577.29 | 634.92 | Stage2 pullback-breakout RSI=61 vol=3.0x ATR=14.84 |
| Stop hit — per-position SL triggered | 2025-07-25 05:30:00 | 629.89 | 577.89 | 635.17 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 692.40 | 599.27 | 665.07 | Stage2 pullback-breakout RSI=68 vol=2.0x ATR=13.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 05:30:00 | 719.32 | 608.83 | 687.46 | T1 booked 50% @ 719.32 |
| Stop hit — per-position SL triggered | 2025-09-24 05:30:00 | 708.20 | 608.83 | 687.46 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 809.00 | 707.64 | 797.89 | Stage2 pullback-breakout RSI=52 vol=3.5x ATR=18.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 05:30:00 | 846.69 | 711.67 | 810.00 | T1 booked 50% @ 846.69 |
| Stop hit — per-position SL triggered | 2026-01-13 05:30:00 | 809.00 | 720.41 | 824.20 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2026-01-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 05:30:00 | 896.75 | 728.47 | 840.62 | Stage2 pullback-breakout RSI=67 vol=2.4x ATR=25.77 |
| Stop hit — per-position SL triggered | 2026-01-27 05:30:00 | 858.10 | 731.40 | 847.15 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-24 05:30:00 | 652.15 | 2025-07-25 05:30:00 | 629.89 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest1 | 2025-09-10 05:30:00 | 692.40 | 2025-09-24 05:30:00 | 719.32 | PARTIAL | 0.50 | 3.89% |
| BUY | retest1 | 2025-09-10 05:30:00 | 692.40 | 2025-09-24 05:30:00 | 708.20 | STOP_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2025-12-30 05:30:00 | 809.00 | 2026-01-02 05:30:00 | 846.69 | PARTIAL | 0.50 | 4.66% |
| BUY | retest1 | 2025-12-30 05:30:00 | 809.00 | 2026-01-13 05:30:00 | 809.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-22 05:30:00 | 896.75 | 2026-01-27 05:30:00 | 858.10 | STOP_HIT | 1.00 | -4.31% |
