# Wipro Ltd. (WIPRO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 197.88
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 5
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 1.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.09% | -0.7% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.09% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.16% | 1.7% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.16% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.06% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 230.40 | 229.08 | 0.00 | ORB-long ORB[227.20,229.98] vol=1.6x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 231.19 | 229.35 | 0.00 | T1 1.5R @ 231.19 |
| Target hit | 2026-02-10 15:20:00 | 231.29 | 230.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 230.60 | 231.47 | 0.00 | ORB-short ORB[231.46,233.00] vol=2.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-02-11 10:40:00 | 231.04 | 231.41 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 216.92 | 214.92 | 0.00 | ORB-long ORB[212.63,215.50] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-02-17 10:25:00 | 216.12 | 215.23 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 213.04 | 214.33 | 0.00 | ORB-short ORB[213.59,216.40] vol=2.3x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:45:00 | 211.90 | 213.60 | 0.00 | T1 1.5R @ 211.90 |
| Target hit | 2026-02-18 15:00:00 | 211.94 | 211.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 211.00 | 210.20 | 0.00 | ORB-long ORB[208.25,210.00] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-02-20 12:55:00 | 210.39 | 210.41 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:30:00 | 205.32 | 207.76 | 0.00 | ORB-short ORB[209.54,211.74] vol=1.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:45:00 | 204.37 | 207.36 | 0.00 | T1 1.5R @ 204.37 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 205.32 | 207.11 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 200.93 | 201.53 | 0.00 | ORB-short ORB[201.10,203.29] vol=2.4x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:40:00 | 199.89 | 201.20 | 0.00 | T1 1.5R @ 199.89 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 200.93 | 201.18 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 10:50:00 | 196.14 | 194.50 | 0.00 | ORB-long ORB[193.03,195.40] vol=2.0x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-03-09 10:55:00 | 195.53 | 194.68 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:00:00 | 194.05 | 196.12 | 0.00 | ORB-short ORB[196.24,198.29] vol=2.7x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 194.68 | 195.91 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 205.15 | 203.84 | 0.00 | ORB-long ORB[202.48,203.90] vol=2.3x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-04-21 10:50:00 | 204.79 | 203.98 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 199.37 | 200.88 | 0.00 | ORB-short ORB[201.11,202.75] vol=2.4x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:15:00 | 198.59 | 200.06 | 0.00 | T1 1.5R @ 198.59 |
| Stop hit — per-position SL triggered | 2026-04-24 11:05:00 | 199.37 | 199.54 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 201.90 | 200.91 | 0.00 | ORB-long ORB[200.05,201.80] vol=1.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-04-30 10:00:00 | 201.42 | 201.36 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 197.43 | 198.10 | 0.00 | ORB-short ORB[197.49,198.94] vol=1.8x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 197.83 | 198.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:05:00 | 230.40 | 2026-02-10 10:15:00 | 231.19 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-10 10:05:00 | 230.40 | 2026-02-10 15:20:00 | 231.29 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-11 10:35:00 | 230.60 | 2026-02-11 10:40:00 | 231.04 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-17 10:05:00 | 216.92 | 2026-02-17 10:25:00 | 216.12 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 09:30:00 | 213.04 | 2026-02-18 09:45:00 | 211.90 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-18 09:30:00 | 213.04 | 2026-02-18 15:00:00 | 211.94 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-20 10:55:00 | 211.00 | 2026-02-20 12:55:00 | 210.39 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-23 10:30:00 | 205.32 | 2026-02-23 10:45:00 | 204.37 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-23 10:30:00 | 205.32 | 2026-02-23 11:05:00 | 205.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 200.93 | 2026-02-24 09:40:00 | 199.89 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-24 09:30:00 | 200.93 | 2026-02-24 09:45:00 | 200.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-09 10:50:00 | 196.14 | 2026-03-09 10:55:00 | 195.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-16 11:00:00 | 194.05 | 2026-03-16 11:15:00 | 194.68 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 10:45:00 | 205.15 | 2026-04-21 10:50:00 | 204.79 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-04-24 09:45:00 | 199.37 | 2026-04-24 10:15:00 | 198.59 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-24 09:45:00 | 199.37 | 2026-04-24 11:05:00 | 199.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-30 09:30:00 | 201.90 | 2026-04-30 10:00:00 | 201.42 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-08 09:30:00 | 197.43 | 2026-05-08 09:35:00 | 197.83 | STOP_HIT | 1.00 | -0.20% |
