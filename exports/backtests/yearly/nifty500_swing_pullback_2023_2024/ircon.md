# IRCON International Ltd. (IRCON)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2024-09-03 05:30:00 (496 bars)
- **Last close:** 256.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 10.85% / 8.11%
- **Sum % (uncompounded):** 86.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 10.85% | 86.8% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 10.85% | 86.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 3 | 3 | 10.85% | 86.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 05:30:00 | 83.65 | 65.43 | 82.40 | Stage2 pullback-breakout RSI=55 vol=1.5x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 05:30:00 | 88.82 | 66.07 | 83.64 | T1 booked 50% @ 88.82 |
| Target hit | 2023-10-09 05:30:00 | 132.90 | 89.56 | 138.03 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 05:30:00 | 149.65 | 91.54 | 138.83 | Stage2 pullback-breakout RSI=62 vol=4.9x ATR=7.32 |
| Stop hit — per-position SL triggered | 2023-10-25 05:30:00 | 138.66 | 95.49 | 144.00 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-12-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 05:30:00 | 172.05 | 114.22 | 163.80 | Stage2 pullback-breakout RSI=61 vol=3.2x ATR=7.20 |
| Stop hit — per-position SL triggered | 2023-12-21 05:30:00 | 161.25 | 117.01 | 166.59 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-01-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 05:30:00 | 187.10 | 121.31 | 170.71 | Stage2 pullback-breakout RSI=68 vol=3.0x ATR=7.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 05:30:00 | 202.89 | 126.57 | 181.88 | T1 booked 50% @ 202.89 |
| Target hit | 2024-02-09 05:30:00 | 222.35 | 143.67 | 222.56 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 05:30:00 | 241.45 | 174.43 | 225.49 | Stage2 pullback-breakout RSI=63 vol=3.2x ATR=9.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 05:30:00 | 261.04 | 176.66 | 231.91 | T1 booked 50% @ 261.04 |
| Stop hit — per-position SL triggered | 2024-05-07 05:30:00 | 241.45 | 180.05 | 237.19 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-17 05:30:00 | 83.65 | 2023-07-20 05:30:00 | 88.82 | PARTIAL | 0.50 | 6.19% |
| BUY | retest1 | 2023-07-17 05:30:00 | 83.65 | 2023-10-09 05:30:00 | 132.90 | TARGET_HIT | 0.50 | 58.88% |
| BUY | retest1 | 2023-10-13 05:30:00 | 149.65 | 2023-10-25 05:30:00 | 138.66 | STOP_HIT | 1.00 | -7.34% |
| BUY | retest1 | 2023-12-14 05:30:00 | 172.05 | 2023-12-21 05:30:00 | 161.25 | STOP_HIT | 1.00 | -6.28% |
| BUY | retest1 | 2024-01-03 05:30:00 | 187.10 | 2024-01-15 05:30:00 | 202.89 | PARTIAL | 0.50 | 8.44% |
| BUY | retest1 | 2024-01-03 05:30:00 | 187.10 | 2024-02-09 05:30:00 | 222.35 | TARGET_HIT | 0.50 | 18.84% |
| BUY | retest1 | 2024-04-24 05:30:00 | 241.45 | 2024-04-29 05:30:00 | 261.04 | PARTIAL | 0.50 | 8.11% |
| BUY | retest1 | 2024-04-24 05:30:00 | 241.45 | 2024-05-07 05:30:00 | 241.45 | STOP_HIT | 0.50 | 0.00% |
