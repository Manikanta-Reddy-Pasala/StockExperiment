# General Insurance Corporation of India (GICRE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 394.05
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
| ENTRY1 | 25 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 20
- **Target hits / Stop hits / Partials:** 5 / 20 / 10
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 7.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 12 | 48.0% | 4 | 13 | 8 | 0.33% | 8.4% |
| BUY @ 2nd Alert (retest1) | 25 | 12 | 48.0% | 4 | 13 | 8 | 0.33% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.06% | -0.6% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.06% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 35 | 15 | 42.9% | 5 | 20 | 10 | 0.22% | 7.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:45:00 | 387.10 | 384.64 | 0.00 | ORB-long ORB[382.30,385.50] vol=2.4x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:50:00 | 388.83 | 385.95 | 0.00 | T1 1.5R @ 388.83 |
| Target hit | 2026-02-10 15:20:00 | 398.00 | 394.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 390.00 | 388.39 | 0.00 | ORB-long ORB[384.65,388.75] vol=3.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-02-17 11:05:00 | 388.95 | 388.54 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 385.80 | 387.66 | 0.00 | ORB-short ORB[386.10,389.75] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2026-02-18 12:45:00 | 386.80 | 387.07 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 387.00 | 385.39 | 0.00 | ORB-long ORB[382.15,386.20] vol=2.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-02-23 10:10:00 | 385.91 | 386.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 390.20 | 389.56 | 0.00 | ORB-long ORB[386.00,388.65] vol=2.2x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:15:00 | 391.79 | 390.68 | 0.00 | T1 1.5R @ 391.79 |
| Target hit | 2026-02-25 10:30:00 | 390.50 | 390.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 394.80 | 393.61 | 0.00 | ORB-long ORB[390.45,394.00] vol=3.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:35:00 | 396.22 | 394.45 | 0.00 | T1 1.5R @ 396.22 |
| Stop hit — per-position SL triggered | 2026-02-26 09:40:00 | 394.80 | 394.52 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 383.50 | 385.21 | 0.00 | ORB-short ORB[386.30,388.75] vol=7.4x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 384.34 | 385.06 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 366.55 | 368.56 | 0.00 | ORB-short ORB[367.00,370.25] vol=1.5x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:35:00 | 365.04 | 367.96 | 0.00 | T1 1.5R @ 365.04 |
| Stop hit — per-position SL triggered | 2026-03-06 13:00:00 | 366.55 | 367.91 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 364.35 | 365.15 | 0.00 | ORB-short ORB[364.75,369.00] vol=6.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-03-10 12:55:00 | 365.15 | 365.01 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 371.70 | 369.66 | 0.00 | ORB-long ORB[365.25,369.95] vol=2.8x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-03-11 10:00:00 | 370.45 | 370.71 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 365.85 | 366.49 | 0.00 | ORB-short ORB[366.15,368.35] vol=2.7x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-03-13 11:00:00 | 366.98 | 366.47 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:25:00 | 353.05 | 356.76 | 0.00 | ORB-short ORB[355.45,359.85] vol=2.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-03-16 10:30:00 | 354.47 | 356.47 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:50:00 | 368.90 | 364.17 | 0.00 | ORB-long ORB[359.15,364.40] vol=5.4x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 12:05:00 | 371.10 | 367.61 | 0.00 | T1 1.5R @ 371.10 |
| Stop hit — per-position SL triggered | 2026-03-17 15:00:00 | 368.90 | 369.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-03-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:25:00 | 370.90 | 370.01 | 0.00 | ORB-long ORB[366.00,370.00] vol=2.1x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:40:00 | 372.64 | 370.37 | 0.00 | T1 1.5R @ 372.64 |
| Target hit | 2026-03-18 14:35:00 | 373.10 | 373.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 368.85 | 367.36 | 0.00 | ORB-long ORB[364.65,367.55] vol=2.2x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-03-19 11:40:00 | 367.91 | 367.56 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:15:00 | 365.00 | 363.52 | 0.00 | ORB-long ORB[359.05,364.35] vol=6.9x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-03-25 13:10:00 | 364.01 | 364.07 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 09:40:00 | 387.00 | 390.87 | 0.00 | ORB-short ORB[390.00,395.65] vol=3.8x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:50:00 | 384.22 | 388.47 | 0.00 | T1 1.5R @ 384.22 |
| Target hit | 2026-04-08 10:05:00 | 386.85 | 385.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2026-04-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:05:00 | 412.00 | 406.76 | 0.00 | ORB-long ORB[400.00,403.00] vol=7.9x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:10:00 | 415.31 | 411.00 | 0.00 | T1 1.5R @ 415.31 |
| Stop hit — per-position SL triggered | 2026-04-16 10:20:00 | 412.00 | 411.77 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 401.00 | 398.43 | 0.00 | ORB-long ORB[395.50,399.85] vol=2.1x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-04-22 11:25:00 | 399.70 | 399.02 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 401.00 | 398.43 | 0.00 | ORB-long ORB[396.25,399.25] vol=2.8x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 399.66 | 398.58 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 396.25 | 398.25 | 0.00 | ORB-short ORB[396.55,401.40] vol=1.7x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-04-24 11:40:00 | 397.58 | 397.53 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 400.50 | 398.12 | 0.00 | ORB-long ORB[396.05,400.45] vol=1.6x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:55:00 | 402.04 | 399.49 | 0.00 | T1 1.5R @ 402.04 |
| Stop hit — per-position SL triggered | 2026-04-27 10:00:00 | 400.50 | 399.76 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-04-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:00:00 | 402.50 | 400.59 | 0.00 | ORB-long ORB[398.85,401.85] vol=2.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-04-28 10:40:00 | 401.59 | 401.65 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 403.35 | 401.85 | 0.00 | ORB-long ORB[400.50,402.80] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 402.47 | 402.08 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-05-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:25:00 | 397.50 | 395.13 | 0.00 | ORB-long ORB[392.55,395.75] vol=2.5x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:15:00 | 399.40 | 396.19 | 0.00 | T1 1.5R @ 399.40 |
| Target hit | 2026-05-04 15:20:00 | 411.20 | 403.50 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:45:00 | 387.10 | 2026-02-10 10:50:00 | 388.83 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-10 10:45:00 | 387.10 | 2026-02-10 15:20:00 | 398.00 | TARGET_HIT | 0.50 | 2.82% |
| BUY | retest1 | 2026-02-17 10:40:00 | 390.00 | 2026-02-17 11:05:00 | 388.95 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-18 10:40:00 | 385.80 | 2026-02-18 12:45:00 | 386.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-23 09:30:00 | 387.00 | 2026-02-23 10:10:00 | 385.91 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-25 10:00:00 | 390.20 | 2026-02-25 10:15:00 | 391.79 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-25 10:00:00 | 390.20 | 2026-02-25 10:30:00 | 390.50 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2026-02-26 09:30:00 | 394.80 | 2026-02-26 09:35:00 | 396.22 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-26 09:30:00 | 394.80 | 2026-02-26 09:40:00 | 394.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:55:00 | 383.50 | 2026-02-27 11:05:00 | 384.34 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-06 10:45:00 | 366.55 | 2026-03-06 12:35:00 | 365.04 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-06 10:45:00 | 366.55 | 2026-03-06 13:00:00 | 366.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 11:10:00 | 364.35 | 2026-03-10 12:55:00 | 365.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-03-11 09:30:00 | 371.70 | 2026-03-11 10:00:00 | 370.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-13 10:40:00 | 365.85 | 2026-03-13 11:00:00 | 366.98 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-16 10:25:00 | 353.05 | 2026-03-16 10:30:00 | 354.47 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-17 10:50:00 | 368.90 | 2026-03-17 12:05:00 | 371.10 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-17 10:50:00 | 368.90 | 2026-03-17 15:00:00 | 368.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:25:00 | 370.90 | 2026-03-18 10:40:00 | 372.64 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-18 10:25:00 | 370.90 | 2026-03-18 14:35:00 | 373.10 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2026-03-19 11:00:00 | 368.85 | 2026-03-19 11:40:00 | 367.91 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-25 11:15:00 | 365.00 | 2026-03-25 13:10:00 | 364.01 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-08 09:40:00 | 387.00 | 2026-04-08 09:50:00 | 384.22 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-04-08 09:40:00 | 387.00 | 2026-04-08 10:05:00 | 386.85 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-04-16 10:05:00 | 412.00 | 2026-04-16 10:10:00 | 415.31 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2026-04-16 10:05:00 | 412.00 | 2026-04-16 10:20:00 | 412.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:55:00 | 401.00 | 2026-04-22 11:25:00 | 399.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-23 10:00:00 | 401.00 | 2026-04-23 10:10:00 | 399.66 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-24 09:45:00 | 396.25 | 2026-04-24 11:40:00 | 397.58 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-27 09:45:00 | 400.50 | 2026-04-27 09:55:00 | 402.04 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-27 09:45:00 | 400.50 | 2026-04-27 10:00:00 | 400.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:00:00 | 402.50 | 2026-04-28 10:40:00 | 401.59 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-29 10:15:00 | 403.35 | 2026-04-29 10:20:00 | 402.47 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-04 10:25:00 | 397.50 | 2026-05-04 11:15:00 | 399.40 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-05-04 10:25:00 | 397.50 | 2026-05-04 15:20:00 | 411.20 | TARGET_HIT | 0.50 | 3.45% |
