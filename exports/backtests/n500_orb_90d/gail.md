# GAIL (India) Ltd. (GAIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 166.59
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 6
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 3.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.27% | 4.1% |
| BUY @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.27% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.05% | -0.3% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.05% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 9 | 42.9% | 3 | 12 | 6 | 0.18% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 163.59 | 162.75 | 0.00 | ORB-long ORB[162.10,163.45] vol=1.6x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 12:15:00 | 164.05 | 163.04 | 0.00 | T1 1.5R @ 164.05 |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 163.59 | 163.33 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 168.80 | 168.28 | 0.00 | ORB-long ORB[167.11,168.67] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-02-19 11:40:00 | 168.41 | 168.42 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 169.25 | 168.17 | 0.00 | ORB-long ORB[165.80,168.19] vol=2.1x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-02-20 11:10:00 | 168.77 | 168.50 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 168.15 | 167.30 | 0.00 | ORB-long ORB[165.60,167.80] vol=2.7x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 167.84 | 167.35 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 170.17 | 170.45 | 0.00 | ORB-short ORB[170.30,171.20] vol=2.2x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:10:00 | 169.70 | 170.43 | 0.00 | T1 1.5R @ 169.70 |
| Target hit | 2026-02-26 14:30:00 | 169.27 | 169.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 169.70 | 168.60 | 0.00 | ORB-long ORB[167.67,169.45] vol=2.9x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:15:00 | 170.30 | 168.94 | 0.00 | T1 1.5R @ 170.30 |
| Stop hit — per-position SL triggered | 2026-02-27 11:30:00 | 169.70 | 168.98 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 155.39 | 155.58 | 0.00 | ORB-short ORB[155.49,157.05] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 155.84 | 155.62 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:10:00 | 149.04 | 148.38 | 0.00 | ORB-long ORB[147.77,148.72] vol=5.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 12:05:00 | 149.56 | 148.60 | 0.00 | T1 1.5R @ 149.56 |
| Target hit | 2026-03-18 15:20:00 | 150.95 | 149.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 137.55 | 137.99 | 0.00 | ORB-short ORB[138.08,139.72] vol=4.7x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-03-27 11:35:00 | 137.94 | 137.95 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 152.40 | 153.22 | 0.00 | ORB-short ORB[152.75,154.98] vol=2.3x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-04-10 13:00:00 | 152.79 | 152.87 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 157.64 | 159.02 | 0.00 | ORB-short ORB[158.48,160.00] vol=1.9x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 158.11 | 158.84 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 161.30 | 160.08 | 0.00 | ORB-long ORB[157.70,159.90] vol=6.2x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:10:00 | 162.02 | 160.29 | 0.00 | T1 1.5R @ 162.02 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 161.30 | 160.78 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 162.58 | 161.57 | 0.00 | ORB-long ORB[160.65,162.10] vol=1.6x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:50:00 | 163.28 | 162.11 | 0.00 | T1 1.5R @ 163.28 |
| Target hit | 2026-04-22 15:20:00 | 166.14 | 164.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 166.28 | 165.56 | 0.00 | ORB-long ORB[164.75,166.23] vol=1.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 165.74 | 165.64 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 167.61 | 166.91 | 0.00 | ORB-long ORB[165.90,167.40] vol=2.2x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-05-08 10:50:00 | 167.25 | 166.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 11:15:00 | 163.59 | 2026-02-12 12:15:00 | 164.05 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-12 11:15:00 | 163.59 | 2026-02-12 13:15:00 | 163.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 10:40:00 | 168.80 | 2026-02-19 11:40:00 | 168.41 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-20 10:35:00 | 169.25 | 2026-02-20 11:10:00 | 168.77 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-24 11:00:00 | 168.15 | 2026-02-24 11:15:00 | 167.84 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-26 11:00:00 | 170.17 | 2026-02-26 11:10:00 | 169.70 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-26 11:00:00 | 170.17 | 2026-02-26 14:30:00 | 169.27 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-27 10:40:00 | 169.70 | 2026-02-27 11:15:00 | 170.30 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-27 10:40:00 | 169.70 | 2026-02-27 11:30:00 | 169.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:55:00 | 155.39 | 2026-03-05 11:00:00 | 155.84 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-18 11:10:00 | 149.04 | 2026-03-18 12:05:00 | 149.56 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-03-18 11:10:00 | 149.04 | 2026-03-18 15:20:00 | 150.95 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2026-03-27 11:10:00 | 137.55 | 2026-03-27 11:35:00 | 137.94 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-10 10:45:00 | 152.40 | 2026-04-10 13:00:00 | 152.79 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-17 09:40:00 | 157.64 | 2026-04-17 09:55:00 | 158.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-21 10:05:00 | 161.30 | 2026-04-21 10:10:00 | 162.02 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-21 10:05:00 | 161.30 | 2026-04-21 11:35:00 | 161.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:30:00 | 162.58 | 2026-04-22 09:50:00 | 163.28 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-22 09:30:00 | 162.58 | 2026-04-22 15:20:00 | 166.14 | TARGET_HIT | 0.50 | 2.19% |
| BUY | retest1 | 2026-05-04 09:30:00 | 166.28 | 2026-05-04 09:50:00 | 165.74 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-08 10:30:00 | 167.61 | 2026-05-08 10:50:00 | 167.25 | STOP_HIT | 1.00 | -0.22% |
