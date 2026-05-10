# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 139.77
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 2.67% / 3.83%
- **Sum % (uncompounded):** 21.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.67% | 21.4% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.67% | 21.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.67% | 21.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 05:30:00 | 121.09 | 111.81 | 118.65 | Stage2 pullback-breakout RSI=56 vol=1.5x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-07-08 05:30:00 | 116.06 | 112.12 | 118.37 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 117.49 | 112.61 | 113.49 | Stage2 pullback-breakout RSI=62 vol=2.8x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 05:30:00 | 121.99 | 113.25 | 117.40 | T1 booked 50% @ 121.99 |
| Stop hit — per-position SL triggered | 2025-09-24 05:30:00 | 120.99 | 113.25 | 117.40 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 05:30:00 | 124.63 | 113.61 | 118.79 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-10-16 05:30:00 | 125.42 | 114.77 | 123.22 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-10-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 05:30:00 | 129.77 | 115.00 | 123.84 | Stage2 pullback-breakout RSI=65 vol=2.8x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 05:30:00 | 136.07 | 115.35 | 125.50 | T1 booked 50% @ 136.07 |
| Target hit | 2025-12-03 05:30:00 | 140.30 | 122.45 | 144.33 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-01-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 05:30:00 | 152.87 | 127.89 | 146.21 | Stage2 pullback-breakout RSI=66 vol=2.2x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 05:30:00 | 160.66 | 128.53 | 148.78 | T1 booked 50% @ 160.66 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 152.87 | 131.36 | 156.17 | SL hit (bars_held=11) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-01 05:30:00 | 121.09 | 2025-07-08 05:30:00 | 116.06 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest1 | 2025-09-10 05:30:00 | 117.49 | 2025-09-24 05:30:00 | 121.99 | PARTIAL | 0.50 | 3.83% |
| BUY | retest1 | 2025-09-10 05:30:00 | 117.49 | 2025-09-24 05:30:00 | 120.99 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest1 | 2025-10-01 05:30:00 | 124.63 | 2025-10-16 05:30:00 | 125.42 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest1 | 2025-10-20 05:30:00 | 129.77 | 2025-10-23 05:30:00 | 136.07 | PARTIAL | 0.50 | 4.85% |
| BUY | retest1 | 2025-10-20 05:30:00 | 129.77 | 2025-12-03 05:30:00 | 140.30 | TARGET_HIT | 0.50 | 8.11% |
| BUY | retest1 | 2026-01-14 05:30:00 | 152.87 | 2026-01-19 05:30:00 | 160.66 | PARTIAL | 0.50 | 5.09% |
| BUY | retest1 | 2026-01-14 05:30:00 | 152.87 | 2026-02-01 05:30:00 | 152.87 | STOP_HIT | 0.50 | 0.00% |
