# Vishal Mega Mart Ltd. (VMM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 124.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 3 / 4 / 3
- **Avg / median % per leg:** 0.15% / 0.29%
- **Sum % (uncompounded):** 1.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.30% | 1.8% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.30% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.08% | -0.3% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.08% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 6 | 60.0% | 3 | 4 | 3 | 0.15% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 120.64 | 120.26 | 0.00 | ORB-long ORB[119.51,120.32] vol=7.4x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 120.40 | 120.27 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 112.66 | 113.28 | 0.00 | ORB-short ORB[113.20,114.28] vol=2.0x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:20:00 | 112.12 | 113.14 | 0.00 | T1 1.5R @ 112.12 |
| Target hit | 2026-03-06 15:20:00 | 112.55 | 112.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:05:00 | 105.07 | 106.79 | 0.00 | ORB-short ORB[107.45,108.99] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 105.62 | 106.27 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:10:00 | 117.04 | 117.69 | 0.00 | ORB-short ORB[117.49,118.93] vol=2.0x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 117.47 | 117.64 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:35:00 | 121.90 | 120.42 | 0.00 | ORB-long ORB[118.51,120.04] vol=2.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-21 10:50:00 | 121.35 | 120.50 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 126.08 | 124.69 | 0.00 | ORB-long ORB[123.61,124.85] vol=3.0x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:00:00 | 126.83 | 125.51 | 0.00 | T1 1.5R @ 126.83 |
| Target hit | 2026-04-23 11:30:00 | 126.44 | 126.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-05-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:05:00 | 124.35 | 124.01 | 0.00 | ORB-long ORB[122.51,124.23] vol=2.0x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 14:20:00 | 124.93 | 124.21 | 0.00 | T1 1.5R @ 124.93 |
| Target hit | 2026-05-04 15:20:00 | 125.73 | 124.35 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:55:00 | 120.64 | 2026-02-17 10:00:00 | 120.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-06 10:55:00 | 112.66 | 2026-03-06 11:20:00 | 112.12 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-06 10:55:00 | 112.66 | 2026-03-06 15:20:00 | 112.55 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2026-04-01 11:05:00 | 105.07 | 2026-04-01 12:15:00 | 105.62 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-04-16 10:10:00 | 117.04 | 2026-04-16 10:30:00 | 117.47 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-21 10:35:00 | 121.90 | 2026-04-21 10:50:00 | 121.35 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-23 09:45:00 | 126.08 | 2026-04-23 10:00:00 | 126.83 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-23 09:45:00 | 126.08 | 2026-04-23 11:30:00 | 126.44 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2026-05-04 11:05:00 | 124.35 | 2026-05-04 14:20:00 | 124.93 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-05-04 11:05:00 | 124.35 | 2026-05-04 15:20:00 | 125.73 | TARGET_HIT | 0.50 | 1.11% |
