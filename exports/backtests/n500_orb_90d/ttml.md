# Tata Teleservices (Maharashtra) Ltd. (TTML)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 44.09
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 4
- **Avg / median % per leg:** 0.46% / 0.09%
- **Sum % (uncompounded):** 5.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.09% | -0.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.09% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.73% | 5.8% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.73% | 5.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.46% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 45.26 | 44.42 | 0.00 | ORB-long ORB[43.82,44.40] vol=5.1x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-02-09 10:45:00 | 44.99 | 44.64 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 43.88 | 44.11 | 0.00 | ORB-short ORB[43.96,44.48] vol=1.9x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-02-13 09:50:00 | 44.00 | 44.04 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 42.10 | 42.26 | 0.00 | ORB-short ORB[42.19,42.67] vol=1.5x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:50:00 | 41.93 | 42.22 | 0.00 | T1 1.5R @ 41.93 |
| Stop hit — per-position SL triggered | 2026-02-25 15:00:00 | 42.10 | 42.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 41.77 | 42.07 | 0.00 | ORB-short ORB[41.85,42.39] vol=2.1x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 12:20:00 | 41.57 | 41.95 | 0.00 | T1 1.5R @ 41.57 |
| Stop hit — per-position SL triggered | 2026-02-26 14:40:00 | 41.77 | 41.89 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 37.92 | 38.25 | 0.00 | ORB-short ORB[38.09,38.52] vol=3.1x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:15:00 | 37.64 | 38.00 | 0.00 | T1 1.5R @ 37.64 |
| Target hit | 2026-03-20 15:20:00 | 36.04 | 36.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 44.55 | 44.03 | 0.00 | ORB-long ORB[43.61,44.04] vol=3.9x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-04-22 09:35:00 | 44.36 | 44.09 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 43.21 | 42.96 | 0.00 | ORB-long ORB[42.63,43.00] vol=2.3x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:40:00 | 43.46 | 43.07 | 0.00 | T1 1.5R @ 43.46 |
| Target hit | 2026-05-05 09:55:00 | 43.25 | 43.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 43.92 | 44.23 | 0.00 | ORB-short ORB[44.13,44.79] vol=1.6x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 44.12 | 44.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 45.26 | 2026-02-09 10:45:00 | 44.99 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2026-02-13 09:30:00 | 43.88 | 2026-02-13 09:50:00 | 44.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-25 11:00:00 | 42.10 | 2026-02-25 11:50:00 | 41.93 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-25 11:00:00 | 42.10 | 2026-02-25 15:00:00 | 42.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 10:40:00 | 41.77 | 2026-02-26 12:20:00 | 41.57 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-26 10:40:00 | 41.77 | 2026-02-26 14:40:00 | 41.77 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 09:35:00 | 37.92 | 2026-03-20 10:15:00 | 37.64 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2026-03-20 09:35:00 | 37.92 | 2026-03-20 15:20:00 | 36.04 | TARGET_HIT | 0.50 | 4.96% |
| BUY | retest1 | 2026-04-22 09:30:00 | 44.55 | 2026-04-22 09:35:00 | 44.36 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-05 09:30:00 | 43.21 | 2026-05-05 09:40:00 | 43.46 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-05-05 09:30:00 | 43.21 | 2026-05-05 09:55:00 | 43.25 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-05-08 09:40:00 | 43.92 | 2026-05-08 09:55:00 | 44.12 | STOP_HIT | 1.00 | -0.46% |
