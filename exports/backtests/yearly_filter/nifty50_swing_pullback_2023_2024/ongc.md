# ONGC (ONGC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 279.20
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 2
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 4.02% / 2.70%
- **Sum % (uncompounded):** 36.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 1 | 5 | 3 | 4.02% | 36.1% |
| BUY @ 2nd Alert (retest1) | 9 | 7 | 77.8% | 1 | 5 | 3 | 4.02% | 36.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 7 | 77.8% | 1 | 5 | 3 | 4.02% | 36.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 00:00:00 | 178.65 | 155.32 | 172.16 | Stage2 pullback-breakout RSI=69 vol=1.5x ATR=3.12 |
| Stop hit — per-position SL triggered | 2023-08-24 00:00:00 | 173.98 | 157.13 | 174.50 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 181.75 | 158.25 | 175.40 | Stage2 pullback-breakout RSI=67 vol=3.1x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 00:00:00 | 188.23 | 160.42 | 180.35 | T1 booked 50% @ 188.23 |
| Stop hit — per-position SL triggered | 2023-09-15 00:00:00 | 186.65 | 160.68 | 180.95 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 202.05 | 172.48 | 193.66 | Stage2 pullback-breakout RSI=64 vol=2.6x ATR=4.50 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 195.29 | 173.54 | 195.57 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 201.05 | 174.67 | 196.06 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 00:00:00 | 210.36 | 175.44 | 197.34 | T1 booked 50% @ 210.36 |
| Stop hit — per-position SL triggered | 2024-01-01 00:00:00 | 205.35 | 177.47 | 201.50 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 214.65 | 178.44 | 203.76 | Stage2 pullback-breakout RSI=70 vol=1.8x ATR=5.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 224.80 | 181.17 | 211.81 | T1 booked 50% @ 224.80 |
| Target hit | 2024-02-29 00:00:00 | 264.60 | 202.65 | 264.74 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-15 00:00:00 | 279.85 | 219.36 | 269.23 | Stage2 pullback-breakout RSI=60 vol=3.9x ATR=8.08 |
| Stop hit — per-position SL triggered | 2024-04-30 00:00:00 | 282.85 | 225.10 | 276.21 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-10 00:00:00 | 178.65 | 2023-08-24 00:00:00 | 173.98 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest1 | 2023-09-01 00:00:00 | 181.75 | 2023-09-14 00:00:00 | 188.23 | PARTIAL | 0.50 | 3.56% |
| BUY | retest1 | 2023-09-01 00:00:00 | 181.75 | 2023-09-15 00:00:00 | 186.65 | STOP_HIT | 0.50 | 2.70% |
| BUY | retest1 | 2023-12-04 00:00:00 | 202.05 | 2023-12-08 00:00:00 | 195.29 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest1 | 2023-12-15 00:00:00 | 201.05 | 2023-12-20 00:00:00 | 210.36 | PARTIAL | 0.50 | 4.63% |
| BUY | retest1 | 2023-12-15 00:00:00 | 201.05 | 2024-01-01 00:00:00 | 205.35 | STOP_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2024-01-04 00:00:00 | 214.65 | 2024-01-15 00:00:00 | 224.80 | PARTIAL | 0.50 | 4.73% |
| BUY | retest1 | 2024-01-04 00:00:00 | 214.65 | 2024-02-29 00:00:00 | 264.60 | TARGET_HIT | 0.50 | 23.27% |
| BUY | retest1 | 2024-04-15 00:00:00 | 279.85 | 2024-04-30 00:00:00 | 282.85 | STOP_HIT | 1.00 | 1.07% |
