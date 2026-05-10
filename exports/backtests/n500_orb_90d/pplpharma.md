# Piramal Pharma Ltd. (PPLPHARMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 179.58
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 6
- **Target hits / Stop hits / Partials:** 5 / 6 / 6
- **Avg / median % per leg:** 0.96% / 0.50%
- **Sum % (uncompounded):** 16.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 9 | 69.2% | 4 | 4 | 5 | 1.26% | 16.3% |
| BUY @ 2nd Alert (retest1) | 13 | 9 | 69.2% | 4 | 4 | 5 | 1.26% | 16.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 11 | 64.7% | 5 | 6 | 6 | 0.96% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 161.90 | 162.51 | 0.00 | ORB-short ORB[162.11,164.39] vol=1.6x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-02-12 11:35:00 | 162.35 | 162.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 167.25 | 166.59 | 0.00 | ORB-long ORB[165.20,166.74] vol=1.5x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 166.80 | 166.74 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 162.63 | 161.88 | 0.00 | ORB-long ORB[161.11,162.32] vol=2.3x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:50:00 | 163.45 | 162.80 | 0.00 | T1 1.5R @ 163.45 |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 162.63 | 163.05 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 151.86 | 153.14 | 0.00 | ORB-short ORB[152.40,154.20] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-03-05 10:55:00 | 152.34 | 153.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 09:45:00 | 141.20 | 140.08 | 0.00 | ORB-long ORB[138.87,140.61] vol=1.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:20:00 | 142.21 | 140.93 | 0.00 | T1 1.5R @ 142.21 |
| Target hit | 2026-03-27 12:15:00 | 142.72 | 142.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 145.66 | 144.79 | 0.00 | ORB-long ORB[143.25,145.10] vol=3.7x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 145.18 | 145.24 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 150.70 | 149.90 | 0.00 | ORB-long ORB[148.90,150.40] vol=4.7x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 151.59 | 150.34 | 0.00 | T1 1.5R @ 151.59 |
| Target hit | 2026-04-21 11:30:00 | 150.97 | 151.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 151.71 | 150.97 | 0.00 | ORB-long ORB[150.00,151.29] vol=2.0x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 151.27 | 151.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 159.90 | 160.78 | 0.00 | ORB-short ORB[160.39,161.52] vol=1.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:10:00 | 159.06 | 160.36 | 0.00 | T1 1.5R @ 159.06 |
| Target hit | 2026-05-05 12:20:00 | 159.75 | 159.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 161.41 | 160.43 | 0.00 | ORB-long ORB[159.88,161.34] vol=4.5x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:05:00 | 162.32 | 161.36 | 0.00 | T1 1.5R @ 162.32 |
| Target hit | 2026-05-06 15:20:00 | 164.99 | 164.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-05-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:10:00 | 167.74 | 166.51 | 0.00 | ORB-long ORB[165.10,167.40] vol=2.2x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:35:00 | 169.03 | 167.93 | 0.00 | T1 1.5R @ 169.03 |
| Target hit | 2026-05-07 11:45:00 | 185.52 | 185.73 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:15:00 | 161.90 | 2026-02-12 11:35:00 | 162.35 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-19 09:30:00 | 167.25 | 2026-02-19 09:40:00 | 166.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-26 09:35:00 | 162.63 | 2026-02-26 09:50:00 | 163.45 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-26 09:35:00 | 162.63 | 2026-02-26 10:15:00 | 162.63 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:50:00 | 151.86 | 2026-03-05 10:55:00 | 152.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-27 09:45:00 | 141.20 | 2026-03-27 10:20:00 | 142.21 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-03-27 09:45:00 | 141.20 | 2026-03-27 12:15:00 | 142.72 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2026-04-10 09:30:00 | 145.66 | 2026-04-10 10:05:00 | 145.18 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 09:30:00 | 150.70 | 2026-04-21 09:35:00 | 151.59 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-21 09:30:00 | 150.70 | 2026-04-21 11:30:00 | 150.97 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-04-22 09:40:00 | 151.71 | 2026-04-22 09:50:00 | 151.27 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-05 09:55:00 | 159.90 | 2026-05-05 10:10:00 | 159.06 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-05 09:55:00 | 159.90 | 2026-05-05 12:20:00 | 159.75 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2026-05-06 10:55:00 | 161.41 | 2026-05-06 11:05:00 | 162.32 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-05-06 10:55:00 | 161.41 | 2026-05-06 15:20:00 | 164.99 | TARGET_HIT | 0.50 | 2.22% |
| BUY | retest1 | 2026-05-07 10:10:00 | 167.74 | 2026-05-07 10:35:00 | 169.03 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-05-07 10:10:00 | 167.74 | 2026-05-07 11:45:00 | 185.52 | TARGET_HIT | 0.50 | 10.60% |
