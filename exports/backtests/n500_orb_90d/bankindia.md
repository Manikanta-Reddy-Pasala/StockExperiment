# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 139.85
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 4
- **Avg / median % per leg:** 0.17% / -0.26%
- **Sum % (uncompounded):** 3.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.40% | 4.4% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.40% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.18% | -1.3% |
| SELL @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.18% | -1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 6 | 33.3% | 2 | 12 | 4 | 0.17% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 161.64 | 162.62 | 0.00 | ORB-short ORB[162.04,164.45] vol=1.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 162.16 | 162.41 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:00:00 | 161.58 | 161.17 | 0.00 | ORB-long ORB[159.56,161.50] vol=1.8x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:15:00 | 162.49 | 161.41 | 0.00 | T1 1.5R @ 162.49 |
| Target hit | 2026-02-16 15:20:00 | 165.76 | 163.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 167.44 | 166.32 | 0.00 | ORB-long ORB[165.00,166.80] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 168.24 | 166.80 | 0.00 | T1 1.5R @ 168.24 |
| Target hit | 2026-02-17 15:20:00 | 170.86 | 168.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 170.70 | 171.51 | 0.00 | ORB-short ORB[170.80,172.50] vol=1.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 171.27 | 171.48 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 175.81 | 174.46 | 0.00 | ORB-long ORB[173.02,174.60] vol=3.7x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:55:00 | 176.67 | 175.31 | 0.00 | T1 1.5R @ 176.67 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 175.81 | 176.08 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:30:00 | 155.20 | 156.38 | 0.00 | ORB-short ORB[156.25,157.70] vol=2.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-03-11 10:35:00 | 155.76 | 156.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:05:00 | 153.60 | 151.67 | 0.00 | ORB-long ORB[149.81,151.88] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 152.95 | 151.74 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 152.89 | 153.46 | 0.00 | ORB-short ORB[152.98,154.65] vol=1.5x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 152.01 | 153.18 | 0.00 | T1 1.5R @ 152.01 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 152.89 | 152.48 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:25:00 | 148.24 | 149.93 | 0.00 | ORB-short ORB[148.52,150.20] vol=1.5x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-03-16 10:30:00 | 148.98 | 149.87 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 147.96 | 147.39 | 0.00 | ORB-long ORB[146.16,147.90] vol=1.7x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 147.35 | 147.65 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 146.97 | 148.10 | 0.00 | ORB-short ORB[147.73,149.47] vol=7.4x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-04-17 11:10:00 | 147.44 | 148.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 151.30 | 150.43 | 0.00 | ORB-long ORB[149.18,150.99] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 150.85 | 150.47 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 141.83 | 141.14 | 0.00 | ORB-long ORB[140.01,141.70] vol=2.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 141.26 | 141.16 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 139.19 | 138.76 | 0.00 | ORB-long ORB[137.63,139.09] vol=2.1x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 138.83 | 138.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 161.64 | 2026-02-13 09:40:00 | 162.16 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-16 10:00:00 | 161.58 | 2026-02-16 10:15:00 | 162.49 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-16 10:00:00 | 161.58 | 2026-02-16 15:20:00 | 165.76 | TARGET_HIT | 0.50 | 2.59% |
| BUY | retest1 | 2026-02-17 10:20:00 | 167.44 | 2026-02-17 10:30:00 | 168.24 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-17 10:20:00 | 167.44 | 2026-02-17 15:20:00 | 170.86 | TARGET_HIT | 0.50 | 2.04% |
| SELL | retest1 | 2026-02-18 09:30:00 | 170.70 | 2026-02-18 09:40:00 | 171.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-24 09:45:00 | 175.81 | 2026-02-24 09:55:00 | 176.67 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-24 09:45:00 | 175.81 | 2026-02-24 11:45:00 | 175.81 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:30:00 | 155.20 | 2026-03-11 10:35:00 | 155.76 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-12 11:05:00 | 153.60 | 2026-03-12 11:15:00 | 152.95 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-13 09:40:00 | 152.89 | 2026-03-13 10:00:00 | 152.01 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-13 09:40:00 | 152.89 | 2026-03-13 11:25:00 | 152.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:25:00 | 148.24 | 2026-03-16 10:30:00 | 148.98 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-10 09:30:00 | 147.96 | 2026-04-10 10:05:00 | 147.35 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-17 11:00:00 | 146.97 | 2026-04-17 11:10:00 | 147.44 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-22 09:45:00 | 151.30 | 2026-04-22 09:50:00 | 150.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-04 09:45:00 | 141.83 | 2026-05-04 09:50:00 | 141.26 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-05 11:00:00 | 139.19 | 2026-05-05 11:15:00 | 138.83 | STOP_HIT | 1.00 | -0.26% |
