# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 144.88
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 0 / 10 / 4
- **Avg / median % per leg:** -0.00% / 0.00%
- **Sum % (uncompounded):** -0.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.27% | -1.4% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.27% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 0 | 5 | 4 | 0.15% | 1.3% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 5 | 4 | 0.15% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 4 | 28.6% | 0 | 10 | 4 | -0.00% | -0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 177.52 | 176.46 | 0.00 | ORB-long ORB[175.20,177.16] vol=2.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 177.02 | 176.68 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 179.64 | 179.32 | 0.00 | ORB-long ORB[176.82,179.30] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-02-11 12:50:00 | 179.21 | 179.49 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 177.87 | 179.80 | 0.00 | ORB-short ORB[179.90,182.25] vol=2.1x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:30:00 | 177.04 | 179.42 | 0.00 | T1 1.5R @ 177.04 |
| Stop hit — per-position SL triggered | 2026-02-12 13:25:00 | 177.87 | 178.39 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 172.64 | 173.28 | 0.00 | ORB-short ORB[173.75,174.95] vol=2.0x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-17 11:35:00 | 172.99 | 173.23 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 177.46 | 176.77 | 0.00 | ORB-long ORB[175.86,177.38] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 177.00 | 177.00 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:00:00 | 176.38 | 177.18 | 0.00 | ORB-short ORB[177.00,177.99] vol=1.8x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:10:00 | 175.75 | 176.99 | 0.00 | T1 1.5R @ 175.75 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 176.38 | 176.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 176.90 | 175.73 | 0.00 | ORB-long ORB[173.67,175.97] vol=3.0x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 176.38 | 175.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 179.20 | 178.26 | 0.00 | ORB-long ORB[176.51,178.67] vol=2.1x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-02-24 09:55:00 | 178.69 | 178.55 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 145.00 | 145.47 | 0.00 | ORB-short ORB[145.04,146.43] vol=1.6x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:55:00 | 144.56 | 145.31 | 0.00 | T1 1.5R @ 144.56 |
| Stop hit — per-position SL triggered | 2026-04-16 10:10:00 | 145.00 | 145.25 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 143.73 | 144.64 | 0.00 | ORB-short ORB[144.50,145.60] vol=1.8x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:45:00 | 143.18 | 144.44 | 0.00 | T1 1.5R @ 143.18 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 143.73 | 143.86 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 177.52 | 2026-02-10 09:45:00 | 177.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-11 10:30:00 | 179.64 | 2026-02-11 12:50:00 | 179.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-12 11:15:00 | 177.87 | 2026-02-12 11:30:00 | 177.04 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-12 11:15:00 | 177.87 | 2026-02-12 13:25:00 | 177.87 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 11:05:00 | 172.64 | 2026-02-17 11:35:00 | 172.99 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-18 09:30:00 | 177.46 | 2026-02-18 09:45:00 | 177.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 10:00:00 | 176.38 | 2026-02-19 10:10:00 | 175.75 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-19 10:00:00 | 176.38 | 2026-02-19 10:40:00 | 176.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 11:00:00 | 176.90 | 2026-02-23 11:10:00 | 176.38 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-24 09:35:00 | 179.20 | 2026-02-24 09:55:00 | 178.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-16 09:35:00 | 145.00 | 2026-04-16 09:55:00 | 144.56 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-16 09:35:00 | 145.00 | 2026-04-16 10:10:00 | 145.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:35:00 | 143.73 | 2026-04-24 09:45:00 | 143.18 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 09:35:00 | 143.73 | 2026-04-24 11:20:00 | 143.73 | STOP_HIT | 0.50 | 0.00% |
