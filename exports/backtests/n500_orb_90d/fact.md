# Fertilisers and Chemicals Travancore Ltd. (FACT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 902.80
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.16% / -0.36%
- **Sum % (uncompounded):** 0.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.37% | -1.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.37% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.95% | 1.9% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.95% | 1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.16% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 803.40 | 796.83 | 0.00 | ORB-long ORB[784.85,792.70] vol=3.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 800.23 | 797.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 795.65 | 799.98 | 0.00 | ORB-short ORB[797.20,807.95] vol=2.4x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:45:00 | 793.72 | 799.56 | 0.00 | T1 1.5R @ 793.72 |
| Target hit | 2026-02-19 15:20:00 | 782.40 | 793.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 915.00 | 912.62 | 0.00 | ORB-long ORB[903.80,914.40] vol=15.3x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-04-29 11:05:00 | 911.75 | 912.55 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 912.20 | 906.07 | 0.00 | ORB-long ORB[900.00,908.70] vol=2.1x ATR=3.28 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 908.92 | 907.41 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:55:00 | 803.40 | 2026-02-17 10:00:00 | 800.23 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-19 11:15:00 | 795.65 | 2026-02-19 11:45:00 | 793.72 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-19 11:15:00 | 795.65 | 2026-02-19 15:20:00 | 782.40 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-04-29 11:00:00 | 915.00 | 2026-04-29 11:05:00 | 911.75 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-05 10:00:00 | 912.20 | 2026-05-05 10:05:00 | 908.92 | STOP_HIT | 1.00 | -0.36% |
