# Piramal Pharma Ltd. (PPLPHARMA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-07-09 15:25:00 (3225 bars)
- **Last close:** 201.42
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 10
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 3.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.19% | 2.0% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.19% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 7 | 41.2% | 1 | 10 | 6 | 0.08% | 1.3% |
| SELL @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 1 | 10 | 6 | 0.08% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 12 | 42.9% | 2 | 16 | 10 | 0.12% | 3.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:30:00 | 207.62 | 206.76 | 0.00 | ORB-long ORB[205.18,207.30] vol=1.8x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-05-21 09:55:00 | 206.76 | 206.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:25:00 | 204.99 | 206.41 | 0.00 | ORB-short ORB[205.74,207.67] vol=2.1x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-05-26 10:35:00 | 205.55 | 206.24 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 11:00:00 | 204.03 | 204.86 | 0.00 | ORB-short ORB[204.70,206.40] vol=2.4x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 12:25:00 | 203.46 | 204.46 | 0.00 | T1 1.5R @ 203.46 |
| Target hit | 2025-05-27 15:20:00 | 203.33 | 203.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-06-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:20:00 | 207.86 | 206.36 | 0.00 | ORB-long ORB[205.36,207.40] vol=4.9x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 10:25:00 | 208.83 | 207.06 | 0.00 | T1 1.5R @ 208.83 |
| Stop hit — per-position SL triggered | 2025-06-02 10:30:00 | 207.86 | 207.22 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 203.11 | 203.77 | 0.00 | ORB-short ORB[203.80,204.80] vol=2.7x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:45:00 | 202.37 | 203.29 | 0.00 | T1 1.5R @ 202.37 |
| Stop hit — per-position SL triggered | 2025-06-04 10:40:00 | 203.11 | 202.87 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 11:15:00 | 208.14 | 206.27 | 0.00 | ORB-long ORB[205.01,206.80] vol=9.4x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 11:20:00 | 208.99 | 206.97 | 0.00 | T1 1.5R @ 208.99 |
| Stop hit — per-position SL triggered | 2025-06-05 13:30:00 | 208.14 | 208.34 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:35:00 | 206.27 | 205.23 | 0.00 | ORB-long ORB[204.00,205.95] vol=2.4x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 10:40:00 | 207.18 | 205.70 | 0.00 | T1 1.5R @ 207.18 |
| Stop hit — per-position SL triggered | 2025-06-12 11:00:00 | 206.27 | 206.35 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:40:00 | 198.60 | 199.84 | 0.00 | ORB-short ORB[199.25,201.83] vol=1.8x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:45:00 | 197.59 | 199.53 | 0.00 | T1 1.5R @ 197.59 |
| Stop hit — per-position SL triggered | 2025-06-16 09:50:00 | 198.60 | 199.40 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 11:00:00 | 198.55 | 199.35 | 0.00 | ORB-short ORB[198.62,200.34] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-06-17 11:10:00 | 199.00 | 199.28 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:15:00 | 192.00 | 192.97 | 0.00 | ORB-short ORB[193.13,194.69] vol=3.5x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:40:00 | 191.31 | 192.54 | 0.00 | T1 1.5R @ 191.31 |
| Stop hit — per-position SL triggered | 2025-06-19 14:10:00 | 192.00 | 192.03 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:10:00 | 195.23 | 192.98 | 0.00 | ORB-long ORB[191.25,192.75] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 194.58 | 193.11 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:40:00 | 197.13 | 196.25 | 0.00 | ORB-long ORB[195.50,197.09] vol=2.7x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 09:50:00 | 198.12 | 196.63 | 0.00 | T1 1.5R @ 198.12 |
| Target hit | 2025-06-24 13:45:00 | 199.60 | 200.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:15:00 | 202.25 | 201.38 | 0.00 | ORB-long ORB[200.11,201.99] vol=1.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-06-25 11:00:00 | 201.69 | 201.56 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 10:00:00 | 201.00 | 202.11 | 0.00 | ORB-short ORB[201.61,203.20] vol=3.9x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-06-27 10:05:00 | 201.60 | 202.05 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:10:00 | 202.81 | 203.92 | 0.00 | ORB-short ORB[203.50,205.37] vol=1.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-06-30 11:25:00 | 203.31 | 203.90 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 11:00:00 | 202.02 | 203.61 | 0.00 | ORB-short ORB[203.78,205.00] vol=3.1x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-07-01 11:05:00 | 202.46 | 203.59 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:20:00 | 203.36 | 204.32 | 0.00 | ORB-short ORB[204.20,205.50] vol=1.9x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:00:00 | 202.56 | 204.08 | 0.00 | T1 1.5R @ 202.56 |
| Stop hit — per-position SL triggered | 2025-07-02 11:20:00 | 203.36 | 203.91 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:45:00 | 199.91 | 201.48 | 0.00 | ORB-short ORB[200.50,202.99] vol=1.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:05:00 | 199.20 | 201.11 | 0.00 | T1 1.5R @ 199.20 |
| Stop hit — per-position SL triggered | 2025-07-08 11:20:00 | 199.91 | 200.94 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-21 09:30:00 | 207.62 | 2025-05-21 09:55:00 | 206.76 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-05-26 10:25:00 | 204.99 | 2025-05-26 10:35:00 | 205.55 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-27 11:00:00 | 204.03 | 2025-05-27 12:25:00 | 203.46 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-05-27 11:00:00 | 204.03 | 2025-05-27 15:20:00 | 203.33 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-06-02 10:20:00 | 207.86 | 2025-06-02 10:25:00 | 208.83 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-02 10:20:00 | 207.86 | 2025-06-02 10:30:00 | 207.86 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:35:00 | 203.11 | 2025-06-04 09:45:00 | 202.37 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-06-04 09:35:00 | 203.11 | 2025-06-04 10:40:00 | 203.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 11:15:00 | 208.14 | 2025-06-05 11:20:00 | 208.99 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-05 11:15:00 | 208.14 | 2025-06-05 13:30:00 | 208.14 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-12 10:35:00 | 206.27 | 2025-06-12 10:40:00 | 207.18 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-12 10:35:00 | 206.27 | 2025-06-12 11:00:00 | 206.27 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-16 09:40:00 | 198.60 | 2025-06-16 09:45:00 | 197.59 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-06-16 09:40:00 | 198.60 | 2025-06-16 09:50:00 | 198.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-17 11:00:00 | 198.55 | 2025-06-17 11:10:00 | 199.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-19 11:15:00 | 192.00 | 2025-06-19 12:40:00 | 191.31 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-19 11:15:00 | 192.00 | 2025-06-19 14:10:00 | 192.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-20 10:10:00 | 195.23 | 2025-06-20 10:15:00 | 194.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-24 09:40:00 | 197.13 | 2025-06-24 09:50:00 | 198.12 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-06-24 09:40:00 | 197.13 | 2025-06-24 13:45:00 | 199.60 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2025-06-25 10:15:00 | 202.25 | 2025-06-25 11:00:00 | 201.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-27 10:00:00 | 201.00 | 2025-06-27 10:05:00 | 201.60 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-30 11:10:00 | 202.81 | 2025-06-30 11:25:00 | 203.31 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-01 11:00:00 | 202.02 | 2025-07-01 11:05:00 | 202.46 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-02 10:20:00 | 203.36 | 2025-07-02 11:00:00 | 202.56 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-02 10:20:00 | 203.36 | 2025-07-02 11:20:00 | 203.36 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 10:45:00 | 199.91 | 2025-07-08 11:05:00 | 199.20 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-08 10:45:00 | 199.91 | 2025-07-08 11:20:00 | 199.91 | STOP_HIT | 0.50 | 0.00% |
