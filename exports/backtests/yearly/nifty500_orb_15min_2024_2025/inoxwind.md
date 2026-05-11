# Inox Wind Ltd. (INOXWIND)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-03-30 15:25:00 (34996 bars)
- **Last close:** 75.05
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
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 17
- **Target hits / Stop hits / Partials:** 1 / 17 / 6
- **Avg / median % per leg:** -0.05% / 0.00%
- **Sum % (uncompounded):** -1.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 4 | 28.6% | 1 | 10 | 3 | -0.11% | -1.5% |
| BUY @ 2nd Alert (retest1) | 14 | 4 | 28.6% | 1 | 10 | 3 | -0.11% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.02% | 0.2% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.02% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 7 | 29.2% | 1 | 17 | 6 | -0.05% | -1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:35:00 | 148.73 | 150.29 | 0.00 | ORB-short ORB[150.24,152.05] vol=2.4x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:20:00 | 147.75 | 149.84 | 0.00 | T1 1.5R @ 147.75 |
| Stop hit — per-position SL triggered | 2024-06-12 11:50:00 | 148.73 | 149.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:40:00 | 137.61 | 138.89 | 0.00 | ORB-short ORB[138.57,139.88] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-06-21 11:05:00 | 138.08 | 138.59 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:35:00 | 138.00 | 139.48 | 0.00 | ORB-short ORB[139.97,141.75] vol=5.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-06-26 11:45:00 | 138.62 | 138.86 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:20:00 | 139.12 | 140.77 | 0.00 | ORB-short ORB[140.66,142.44] vol=1.7x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-06-27 10:35:00 | 139.89 | 140.41 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 241.50 | 242.66 | 0.00 | ORB-short ORB[241.80,244.36] vol=1.6x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:35:00 | 239.87 | 242.14 | 0.00 | T1 1.5R @ 239.87 |
| Stop hit — per-position SL triggered | 2024-09-26 09:50:00 | 241.50 | 241.56 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 242.72 | 239.86 | 0.00 | ORB-long ORB[238.17,241.50] vol=1.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-09-27 10:55:00 | 241.53 | 240.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:55:00 | 216.86 | 215.27 | 0.00 | ORB-long ORB[213.43,216.05] vol=1.9x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-10-11 10:05:00 | 215.86 | 215.39 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 217.42 | 215.84 | 0.00 | ORB-long ORB[213.90,216.86] vol=2.2x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-10-31 10:05:00 | 216.24 | 216.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-11-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:35:00 | 183.59 | 182.50 | 0.00 | ORB-long ORB[181.42,182.85] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-11-27 10:50:00 | 182.91 | 182.57 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-12-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:35:00 | 206.31 | 199.52 | 0.00 | ORB-long ORB[185.76,188.77] vol=6.1x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-12-02 10:40:00 | 204.44 | 200.54 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 203.67 | 202.78 | 0.00 | ORB-long ORB[201.31,203.55] vol=1.8x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-12-06 09:45:00 | 202.84 | 202.82 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-12-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:40:00 | 208.36 | 206.95 | 0.00 | ORB-long ORB[205.03,207.35] vol=3.5x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-12-10 09:45:00 | 207.37 | 207.00 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-12-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 09:50:00 | 204.59 | 204.91 | 0.00 | ORB-short ORB[204.69,206.01] vol=1.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:00:00 | 203.34 | 204.73 | 0.00 | T1 1.5R @ 203.34 |
| Stop hit — per-position SL triggered | 2024-12-11 11:35:00 | 204.59 | 204.99 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 187.02 | 186.06 | 0.00 | ORB-long ORB[185.07,186.89] vol=1.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 186.27 | 186.30 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:15:00 | 183.05 | 184.33 | 0.00 | ORB-short ORB[183.35,185.70] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-01-02 12:00:00 | 183.75 | 184.21 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 186.16 | 185.66 | 0.00 | ORB-long ORB[184.58,185.75] vol=2.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 09:40:00 | 187.38 | 186.12 | 0.00 | T1 1.5R @ 187.38 |
| Stop hit — per-position SL triggered | 2025-01-03 09:50:00 | 186.16 | 186.14 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:25:00 | 161.73 | 160.47 | 0.00 | ORB-long ORB[158.80,160.97] vol=1.8x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:30:00 | 162.66 | 161.23 | 0.00 | T1 1.5R @ 162.66 |
| Stop hit — per-position SL triggered | 2025-04-16 11:00:00 | 161.73 | 162.33 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 167.10 | 166.22 | 0.00 | ORB-long ORB[164.81,166.87] vol=3.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:35:00 | 168.48 | 166.76 | 0.00 | T1 1.5R @ 168.48 |
| Target hit | 2025-05-05 11:30:00 | 167.87 | 168.02 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-06-12 10:35:00 | 148.73 | 2024-06-12 11:20:00 | 147.75 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-06-12 10:35:00 | 148.73 | 2024-06-12 11:50:00 | 148.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 10:40:00 | 137.61 | 2024-06-21 11:05:00 | 138.08 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-26 10:35:00 | 138.00 | 2024-06-26 11:45:00 | 138.62 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-06-27 10:20:00 | 139.12 | 2024-06-27 10:35:00 | 139.89 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-09-26 09:30:00 | 241.50 | 2024-09-26 09:35:00 | 239.87 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-09-26 09:30:00 | 241.50 | 2024-09-26 09:50:00 | 241.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:15:00 | 242.72 | 2024-09-27 10:55:00 | 241.53 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-10-11 09:55:00 | 216.86 | 2024-10-11 10:05:00 | 215.86 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-10-31 09:45:00 | 217.42 | 2024-10-31 10:05:00 | 216.24 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-11-27 10:35:00 | 183.59 | 2024-11-27 10:50:00 | 182.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-02 10:35:00 | 206.31 | 2024-12-02 10:40:00 | 204.44 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest1 | 2024-12-06 09:35:00 | 203.67 | 2024-12-06 09:45:00 | 202.84 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-10 09:40:00 | 208.36 | 2024-12-10 09:45:00 | 207.37 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-12-11 09:50:00 | 204.59 | 2024-12-11 10:00:00 | 203.34 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-12-11 09:50:00 | 204.59 | 2024-12-11 11:35:00 | 204.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 09:35:00 | 187.02 | 2024-12-20 09:45:00 | 186.27 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-02 11:15:00 | 183.05 | 2025-01-02 12:00:00 | 183.75 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-03 09:30:00 | 186.16 | 2025-01-03 09:40:00 | 187.38 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-03 09:30:00 | 186.16 | 2025-01-03 09:50:00 | 186.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-16 10:25:00 | 161.73 | 2025-04-16 10:30:00 | 162.66 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-04-16 10:25:00 | 161.73 | 2025-04-16 11:00:00 | 161.73 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:30:00 | 167.10 | 2025-05-05 09:35:00 | 168.48 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2025-05-05 09:30:00 | 167.10 | 2025-05-05 11:30:00 | 167.87 | TARGET_HIT | 0.50 | 0.46% |
