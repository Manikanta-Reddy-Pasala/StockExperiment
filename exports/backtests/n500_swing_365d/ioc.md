# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 144.69
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
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 1.88% / 3.34%
- **Sum % (uncompounded):** 15.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 1 | 4 | 3 | 1.88% | 15.0% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 4 | 3 | 1.88% | 15.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 1 | 4 | 3 | 1.88% | 15.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 05:30:00 | 149.31 | 143.97 | 145.08 | Stage2 pullback-breakout RSI=65 vol=2.1x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 05:30:00 | 154.29 | 144.25 | 147.16 | T1 booked 50% @ 154.29 |
| Stop hit — per-position SL triggered | 2025-10-14 05:30:00 | 152.81 | 144.83 | 150.33 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 155.20 | 145.46 | 151.76 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 05:30:00 | 160.93 | 145.73 | 153.08 | T1 booked 50% @ 160.93 |
| Target hit | 2025-11-24 05:30:00 | 165.69 | 149.41 | 166.40 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-12-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 05:30:00 | 168.55 | 151.38 | 164.32 | Stage2 pullback-breakout RSI=61 vol=3.0x ATR=3.42 |
| Stop hit — per-position SL triggered | 2025-12-18 05:30:00 | 163.42 | 151.81 | 164.67 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 166.46 | 152.64 | 163.60 | Stage2 pullback-breakout RSI=56 vol=1.9x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-01-06 05:30:00 | 161.44 | 153.14 | 164.17 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2026-02-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 05:30:00 | 172.78 | 154.58 | 162.85 | Stage2 pullback-breakout RSI=67 vol=2.3x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 05:30:00 | 181.35 | 155.69 | 168.66 | T1 booked 50% @ 181.35 |
| Stop hit — per-position SL triggered | 2026-02-17 05:30:00 | 172.78 | 156.51 | 171.20 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-29 05:30:00 | 149.31 | 2025-10-06 05:30:00 | 154.29 | PARTIAL | 0.50 | 3.34% |
| BUY | retest1 | 2025-09-29 05:30:00 | 149.31 | 2025-10-14 05:30:00 | 152.81 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2025-10-27 05:30:00 | 155.20 | 2025-10-29 05:30:00 | 160.93 | PARTIAL | 0.50 | 3.69% |
| BUY | retest1 | 2025-10-27 05:30:00 | 155.20 | 2025-11-24 05:30:00 | 165.69 | TARGET_HIT | 0.50 | 6.76% |
| BUY | retest1 | 2025-12-15 05:30:00 | 168.55 | 2025-12-18 05:30:00 | 163.42 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest1 | 2025-12-31 05:30:00 | 166.46 | 2026-01-06 05:30:00 | 161.44 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest1 | 2026-02-04 05:30:00 | 172.78 | 2026-02-11 05:30:00 | 181.35 | PARTIAL | 0.50 | 4.96% |
| BUY | retest1 | 2026-02-04 05:30:00 | 172.78 | 2026-02-17 05:30:00 | 172.78 | STOP_HIT | 0.50 | 0.00% |
