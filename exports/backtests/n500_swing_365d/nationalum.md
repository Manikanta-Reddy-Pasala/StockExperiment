# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 401.95
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 3 / 1 / 4
- **Avg / median % per leg:** 12.34% / 5.40%
- **Sum % (uncompounded):** 98.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 7 | 87.5% | 3 | 1 | 4 | 12.34% | 98.8% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 3 | 1 | 4 | 12.34% | 98.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 3 | 1 | 4 | 12.34% | 98.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 05:30:00 | 200.53 | 188.27 | 189.87 | Stage2 pullback-breakout RSI=65 vol=4.0x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 05:30:00 | 210.17 | 188.88 | 194.78 | T1 booked 50% @ 210.17 |
| Target hit | 2025-09-24 05:30:00 | 206.09 | 191.70 | 207.06 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 05:30:00 | 213.87 | 192.26 | 206.83 | Stage2 pullback-breakout RSI=61 vol=2.7x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 05:30:00 | 225.41 | 193.55 | 211.62 | T1 booked 50% @ 225.41 |
| Target hit | 2026-02-01 05:30:00 | 354.20 | 249.46 | 361.53 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-03-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 05:30:00 | 373.50 | 270.72 | 356.72 | Stage2 pullback-breakout RSI=59 vol=2.8x ATR=14.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 05:30:00 | 402.37 | 271.97 | 360.46 | T1 booked 50% @ 402.37 |
| Stop hit — per-position SL triggered | 2026-03-16 05:30:00 | 373.50 | 280.04 | 375.59 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2026-03-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 05:30:00 | 386.10 | 288.03 | 373.49 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=17.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 05:30:00 | 420.92 | 297.30 | 393.53 | T1 booked 50% @ 420.92 |
| Target hit | 2026-04-30 05:30:00 | 399.30 | 312.42 | 419.12 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-02 05:30:00 | 200.53 | 2025-09-05 05:30:00 | 210.17 | PARTIAL | 0.50 | 4.81% |
| BUY | retest1 | 2025-09-02 05:30:00 | 200.53 | 2025-09-24 05:30:00 | 206.09 | TARGET_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2025-09-30 05:30:00 | 213.87 | 2025-10-08 05:30:00 | 225.41 | PARTIAL | 0.50 | 5.40% |
| BUY | retest1 | 2025-09-30 05:30:00 | 213.87 | 2026-02-01 05:30:00 | 354.20 | TARGET_HIT | 0.50 | 65.61% |
| BUY | retest1 | 2026-03-04 05:30:00 | 373.50 | 2026-03-05 05:30:00 | 402.37 | PARTIAL | 0.50 | 7.73% |
| BUY | retest1 | 2026-03-04 05:30:00 | 373.50 | 2026-03-16 05:30:00 | 373.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-30 05:30:00 | 386.10 | 2026-04-13 05:30:00 | 420.92 | PARTIAL | 0.50 | 9.02% |
| BUY | retest1 | 2026-03-30 05:30:00 | 386.10 | 2026-04-30 05:30:00 | 399.30 | TARGET_HIT | 0.50 | 3.42% |
