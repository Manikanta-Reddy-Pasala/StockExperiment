# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 120.41
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 4.64% / 4.87%
- **Sum % (uncompounded):** 23.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 4.64% | 23.2% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 4.64% | 23.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 4.64% | 23.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 05:30:00 | 146.79 | 136.06 | 139.05 | Stage2 pullback-breakout RSI=69 vol=6.6x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 05:30:00 | 153.94 | 136.38 | 141.46 | T1 booked 50% @ 153.94 |
| Target hit | 2025-07-29 05:30:00 | 150.39 | 138.56 | 150.67 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 152.98 | 139.49 | 147.48 | Stage2 pullback-breakout RSI=60 vol=2.5x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 05:30:00 | 162.60 | 140.00 | 150.07 | T1 booked 50% @ 162.60 |
| Target hit | 2025-09-22 05:30:00 | 169.55 | 146.04 | 170.00 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-12-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 05:30:00 | 163.28 | 152.09 | 157.55 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=4.26 |
| Stop hit — per-position SL triggered | 2025-12-15 05:30:00 | 161.25 | 153.06 | 160.52 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-08 05:30:00 | 146.79 | 2025-07-10 05:30:00 | 153.94 | PARTIAL | 0.50 | 4.87% |
| BUY | retest1 | 2025-07-08 05:30:00 | 146.79 | 2025-07-29 05:30:00 | 150.39 | TARGET_HIT | 0.50 | 2.45% |
| BUY | retest1 | 2025-08-18 05:30:00 | 152.98 | 2025-08-21 05:30:00 | 162.60 | PARTIAL | 0.50 | 6.29% |
| BUY | retest1 | 2025-08-18 05:30:00 | 152.98 | 2025-09-22 05:30:00 | 169.55 | TARGET_HIT | 0.50 | 10.83% |
| BUY | retest1 | 2025-12-01 05:30:00 | 163.28 | 2025-12-15 05:30:00 | 161.25 | STOP_HIT | 1.00 | -1.24% |
