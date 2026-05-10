# BEL (BEL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 439.70
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
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.39% / -1.08%
- **Sum % (uncompounded):** -1.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.39% | -1.9% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.39% | -1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.39% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 388.05 | 346.67 | 377.39 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=7.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 05:30:00 | 403.74 | 349.26 | 386.59 | T1 booked 50% @ 403.74 |
| Stop hit — per-position SL triggered | 2025-09-24 05:30:00 | 395.45 | 351.97 | 393.58 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 05:30:00 | 426.10 | 365.00 | 411.83 | Stage2 pullback-breakout RSI=63 vol=3.5x ATR=8.27 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 413.69 | 366.49 | 412.65 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 413.80 | 379.19 | 399.43 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=8.39 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 409.35 | 382.54 | 408.61 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2026-03-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 05:30:00 | 453.95 | 396.29 | 439.45 | Stage2 pullback-breakout RSI=61 vol=2.4x ATR=11.77 |
| Stop hit — per-position SL triggered | 2026-03-13 05:30:00 | 436.30 | 400.82 | 447.73 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-10 05:30:00 | 388.05 | 2025-09-17 05:30:00 | 403.74 | PARTIAL | 0.50 | 4.04% |
| BUY | retest1 | 2025-09-10 05:30:00 | 388.05 | 2025-09-24 05:30:00 | 395.45 | STOP_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2025-10-31 05:30:00 | 426.10 | 2025-11-06 05:30:00 | 413.69 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest1 | 2026-01-05 05:30:00 | 413.80 | 2026-01-20 05:30:00 | 409.35 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest1 | 2026-03-02 05:30:00 | 453.95 | 2026-03-13 05:30:00 | 436.30 | STOP_HIT | 1.00 | -3.89% |
