# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
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
- **Avg / median % per leg:** 6.82% / 4.44%
- **Sum % (uncompounded):** 54.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.82% | 54.5% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.82% | 54.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 2 | 3 | 3 | 6.82% | 54.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 99.75 | 87.22 | 95.70 | Stage2 pullback-breakout RSI=60 vol=3.6x ATR=2.83 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 95.50 | 87.87 | 96.78 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 93.75 | 88.95 | 92.63 | Stage2 pullback-breakout RSI=53 vol=1.5x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 00:00:00 | 97.91 | 89.10 | 93.35 | T1 booked 50% @ 97.91 |
| Target hit | 2024-01-23 00:00:00 | 126.75 | 98.51 | 128.04 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 141.10 | 99.30 | 129.94 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 00:00:00 | 155.43 | 101.68 | 137.17 | T1 booked 50% @ 155.43 |
| Target hit | 2024-02-12 00:00:00 | 143.20 | 104.89 | 145.74 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 162.75 | 106.84 | 148.88 | Stage2 pullback-breakout RSI=65 vol=1.9x ATR=9.21 |
| Stop hit — per-position SL triggered | 2024-03-01 00:00:00 | 160.80 | 111.70 | 154.45 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 174.75 | 119.54 | 154.27 | Stage2 pullback-breakout RSI=69 vol=2.9x ATR=7.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 00:00:00 | 189.97 | 125.96 | 172.65 | T1 booked 50% @ 189.97 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 174.75 | 131.60 | 180.26 | SL hit (bars_held=21) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-12 00:00:00 | 99.75 | 2023-10-20 00:00:00 | 95.50 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest1 | 2023-12-01 00:00:00 | 93.75 | 2023-12-05 00:00:00 | 97.91 | PARTIAL | 0.50 | 4.44% |
| BUY | retest1 | 2023-12-01 00:00:00 | 93.75 | 2024-01-23 00:00:00 | 126.75 | TARGET_HIT | 0.50 | 35.20% |
| BUY | retest1 | 2024-01-25 00:00:00 | 141.10 | 2024-02-02 00:00:00 | 155.43 | PARTIAL | 0.50 | 10.15% |
| BUY | retest1 | 2024-01-25 00:00:00 | 141.10 | 2024-02-12 00:00:00 | 143.20 | TARGET_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2024-02-16 00:00:00 | 162.75 | 2024-03-01 00:00:00 | 160.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest1 | 2024-04-03 00:00:00 | 174.75 | 2024-04-22 00:00:00 | 189.97 | PARTIAL | 0.50 | 8.71% |
| BUY | retest1 | 2024-04-03 00:00:00 | 174.75 | 2024-05-07 00:00:00 | 174.75 | STOP_HIT | 0.50 | 0.00% |
