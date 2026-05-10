# NTPC Green Energy Ltd. (NTPCGREEN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 107.55
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 6
- **Target hits / Stop hits / Partials:** 3 / 6 / 6
- **Avg / median % per leg:** 0.98% / 0.26%
- **Sum % (uncompounded):** 14.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.22% | 13.3% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.22% | 13.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 5 | 55.6% | 1 | 4 | 4 | 0.15% | 1.4% |
| SELL @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 4 | 4 | 0.15% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 9 | 60.0% | 3 | 6 | 6 | 0.98% | 14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:05:00 | 89.23 | 88.55 | 0.00 | ORB-long ORB[88.00,88.50] vol=4.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 13:25:00 | 89.59 | 88.98 | 0.00 | T1 1.5R @ 89.59 |
| Target hit | 2026-02-16 15:20:00 | 89.51 | 89.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 89.48 | 89.63 | 0.00 | ORB-short ORB[89.50,90.15] vol=1.6x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:00:00 | 89.20 | 89.53 | 0.00 | T1 1.5R @ 89.20 |
| Stop hit — per-position SL triggered | 2026-02-18 10:20:00 | 89.48 | 89.48 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 89.25 | 89.29 | 0.00 | ORB-short ORB[89.31,89.73] vol=7.7x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:40:00 | 88.98 | 89.23 | 0.00 | T1 1.5R @ 88.98 |
| Target hit | 2026-02-19 13:35:00 | 89.19 | 89.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 89.34 | 89.53 | 0.00 | ORB-short ORB[89.45,90.00] vol=2.1x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-02-24 10:10:00 | 89.57 | 89.49 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 90.56 | 90.13 | 0.00 | ORB-long ORB[89.50,90.40] vol=4.1x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 90.37 | 90.19 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 89.61 | 89.67 | 0.00 | ORB-short ORB[89.65,90.23] vol=5.3x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 12:05:00 | 89.38 | 89.65 | 0.00 | T1 1.5R @ 89.38 |
| Stop hit — per-position SL triggered | 2026-02-27 12:45:00 | 89.61 | 89.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 88.09 | 87.56 | 0.00 | ORB-long ORB[86.65,87.88] vol=2.3x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-03-11 10:35:00 | 87.79 | 87.58 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 86.90 | 86.13 | 0.00 | ORB-long ORB[85.23,86.51] vol=2.2x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:40:00 | 87.49 | 86.52 | 0.00 | T1 1.5R @ 87.49 |
| Target hit | 2026-03-12 14:40:00 | 97.71 | 98.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 109.48 | 111.01 | 0.00 | ORB-short ORB[110.26,111.75] vol=2.7x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 15:05:00 | 108.70 | 109.76 | 0.00 | T1 1.5R @ 108.70 |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 109.48 | 109.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 10:05:00 | 89.23 | 2026-02-16 13:25:00 | 89.59 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-16 10:05:00 | 89.23 | 2026-02-16 15:20:00 | 89.51 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 09:40:00 | 89.48 | 2026-02-18 10:00:00 | 89.20 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 09:40:00 | 89.48 | 2026-02-18 10:20:00 | 89.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 89.25 | 2026-02-19 12:40:00 | 88.98 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-19 11:15:00 | 89.25 | 2026-02-19 13:35:00 | 89.19 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2026-02-24 09:30:00 | 89.34 | 2026-02-24 10:10:00 | 89.57 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-25 10:50:00 | 90.56 | 2026-02-25 11:05:00 | 90.37 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 11:10:00 | 89.61 | 2026-02-27 12:05:00 | 89.38 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-27 11:10:00 | 89.61 | 2026-02-27 12:45:00 | 89.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 10:25:00 | 88.09 | 2026-03-11 10:35:00 | 87.79 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-12 09:30:00 | 86.90 | 2026-03-12 09:40:00 | 87.49 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-12 09:30:00 | 86.90 | 2026-03-12 14:40:00 | 97.71 | TARGET_HIT | 0.50 | 12.44% |
| SELL | retest1 | 2026-05-05 10:35:00 | 109.48 | 2026-05-05 15:05:00 | 108.70 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-05-05 10:35:00 | 109.48 | 2026-05-05 15:15:00 | 109.48 | STOP_HIT | 0.50 | 0.00% |
