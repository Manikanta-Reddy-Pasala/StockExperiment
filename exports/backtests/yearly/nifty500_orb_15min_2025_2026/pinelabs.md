# Pine Labs Ltd. (PINELABS)

## Backtest Summary

- **Window:** 2025-11-14 09:15:00 → 2026-05-08 15:25:00 (8850 bars)
- **Last close:** 196.60
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 6
- **Avg / median % per leg:** 0.06% / -0.27%
- **Sum % (uncompounded):** 1.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.02% | 0.2% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.02% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.10% | 1.3% |
| SELL @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.10% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 9 | 36.0% | 3 | 16 | 6 | 0.06% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:45:00 | 242.17 | 240.27 | 0.00 | ORB-long ORB[239.64,241.66] vol=1.9x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-11-19 10:55:00 | 241.07 | 240.56 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-12-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:50:00 | 237.53 | 239.63 | 0.00 | ORB-short ORB[239.05,241.88] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-12-09 10:05:00 | 238.81 | 239.48 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-12-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:30:00 | 240.94 | 242.13 | 0.00 | ORB-short ORB[243.82,246.08] vol=7.2x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-12-10 10:40:00 | 241.86 | 241.79 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:55:00 | 222.39 | 222.95 | 0.00 | ORB-short ORB[224.14,225.84] vol=2.8x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-12-18 10:10:00 | 223.33 | 222.91 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-12-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:40:00 | 226.91 | 225.38 | 0.00 | ORB-long ORB[223.99,225.93] vol=2.5x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:45:00 | 227.79 | 225.51 | 0.00 | T1 1.5R @ 227.79 |
| Stop hit — per-position SL triggered | 2025-12-19 11:05:00 | 226.91 | 225.94 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-12-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:00:00 | 236.25 | 234.10 | 0.00 | ORB-long ORB[232.54,236.00] vol=5.7x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 11:10:00 | 237.48 | 234.77 | 0.00 | T1 1.5R @ 237.48 |
| Stop hit — per-position SL triggered | 2025-12-23 11:35:00 | 236.25 | 235.07 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-12-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:35:00 | 234.75 | 235.92 | 0.00 | ORB-short ORB[235.60,238.50] vol=1.9x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-12-29 09:40:00 | 235.47 | 235.86 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-12-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:00:00 | 238.73 | 237.21 | 0.00 | ORB-long ORB[235.10,237.50] vol=1.5x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-12-31 10:10:00 | 238.01 | 237.29 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:40:00 | 233.83 | 235.05 | 0.00 | ORB-short ORB[233.86,236.97] vol=2.3x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-01-02 10:45:00 | 234.45 | 235.02 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 240.05 | 238.57 | 0.00 | ORB-long ORB[235.58,237.08] vol=2.4x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-01-06 10:40:00 | 239.14 | 238.97 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-01-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:20:00 | 235.07 | 235.49 | 0.00 | ORB-short ORB[235.45,236.81] vol=1.5x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:05:00 | 234.25 | 235.39 | 0.00 | T1 1.5R @ 234.25 |
| Target hit | 2026-01-08 15:20:00 | 230.23 | 232.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-01-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:20:00 | 213.63 | 214.68 | 0.00 | ORB-short ORB[213.99,216.31] vol=2.2x ATR=0.59 |
| Stop hit — per-position SL triggered | 2026-01-14 10:30:00 | 214.22 | 214.63 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-01-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:20:00 | 234.88 | 235.39 | 0.00 | ORB-short ORB[235.10,238.38] vol=1.8x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-01-22 11:20:00 | 236.42 | 235.25 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-01 10:50:00 | 225.82 | 226.53 | 0.00 | ORB-short ORB[226.70,229.01] vol=2.2x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 12:00:00 | 224.39 | 226.15 | 0.00 | T1 1.5R @ 224.39 |
| Target hit | 2026-02-01 13:55:00 | 224.09 | 224.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2026-02-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:50:00 | 221.99 | 224.28 | 0.00 | ORB-short ORB[223.70,227.00] vol=2.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-02-05 11:00:00 | 222.79 | 224.06 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 210.53 | 213.02 | 0.00 | ORB-short ORB[212.55,215.06] vol=2.6x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 208.97 | 212.73 | 0.00 | T1 1.5R @ 208.97 |
| Stop hit — per-position SL triggered | 2026-02-10 10:25:00 | 210.53 | 212.57 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 205.54 | 203.70 | 0.00 | ORB-long ORB[200.92,203.99] vol=2.5x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-02-17 10:25:00 | 204.55 | 203.85 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 161.14 | 159.78 | 0.00 | ORB-long ORB[158.55,160.58] vol=2.3x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 160.18 | 160.45 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 203.80 | 201.34 | 0.00 | ORB-long ORB[198.38,199.98] vol=2.3x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:40:00 | 205.57 | 202.74 | 0.00 | T1 1.5R @ 205.57 |
| Target hit | 2026-04-28 11:25:00 | 205.15 | 205.67 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-19 10:45:00 | 242.17 | 2025-11-19 10:55:00 | 241.07 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-12-09 09:50:00 | 237.53 | 2025-12-09 10:05:00 | 238.81 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-12-10 10:30:00 | 240.94 | 2025-12-10 10:40:00 | 241.86 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-12-18 09:55:00 | 222.39 | 2025-12-18 10:10:00 | 223.33 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-12-19 10:40:00 | 226.91 | 2025-12-19 10:45:00 | 227.79 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-19 10:40:00 | 226.91 | 2025-12-19 11:05:00 | 226.91 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 11:00:00 | 236.25 | 2025-12-23 11:10:00 | 237.48 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-12-23 11:00:00 | 236.25 | 2025-12-23 11:35:00 | 236.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 09:35:00 | 234.75 | 2025-12-29 09:40:00 | 235.47 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-31 10:00:00 | 238.73 | 2025-12-31 10:10:00 | 238.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-02 10:40:00 | 233.83 | 2026-01-02 10:45:00 | 234.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-06 10:15:00 | 240.05 | 2026-01-06 10:40:00 | 239.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-01-08 10:20:00 | 235.07 | 2026-01-08 11:05:00 | 234.25 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-08 10:20:00 | 235.07 | 2026-01-08 15:20:00 | 230.23 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2026-01-14 10:20:00 | 213.63 | 2026-01-14 10:30:00 | 214.22 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-22 10:20:00 | 234.88 | 2026-01-22 11:20:00 | 236.42 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2026-02-01 10:50:00 | 225.82 | 2026-02-01 12:00:00 | 224.39 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-02-01 10:50:00 | 225.82 | 2026-02-01 13:55:00 | 224.09 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2026-02-05 10:50:00 | 221.99 | 2026-02-05 11:00:00 | 222.79 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-10 10:10:00 | 210.53 | 2026-02-10 10:15:00 | 208.97 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2026-02-10 10:10:00 | 210.53 | 2026-02-10 10:25:00 | 210.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:20:00 | 205.54 | 2026-02-17 10:25:00 | 204.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-16 09:30:00 | 161.14 | 2026-03-16 10:15:00 | 160.18 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-28 09:35:00 | 203.80 | 2026-04-28 09:40:00 | 205.57 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2026-04-28 09:35:00 | 203.80 | 2026-04-28 11:25:00 | 205.15 | TARGET_HIT | 0.50 | 0.66% |
