# Tata Capital Ltd. (TATACAP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 321.70
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 14
- **Target hits / Stop hits / Partials:** 6 / 14 / 11
- **Avg / median % per leg:** 0.18% / 0.29%
- **Sum % (uncompounded):** 5.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.24% | 3.4% |
| BUY @ 2nd Alert (retest1) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.24% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.13% | 2.1% |
| SELL @ 2nd Alert (retest1) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.13% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 31 | 17 | 54.8% | 6 | 14 | 11 | 0.18% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 351.80 | 350.92 | 0.00 | ORB-long ORB[344.25,349.25] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-02-09 15:20:00 | 351.25 | 351.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:10:00 | 351.65 | 356.18 | 0.00 | ORB-short ORB[355.60,358.45] vol=2.9x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-02-12 11:30:00 | 352.62 | 354.76 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 353.05 | 350.97 | 0.00 | ORB-long ORB[347.60,352.50] vol=2.2x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:15:00 | 354.18 | 352.88 | 0.00 | T1 1.5R @ 354.18 |
| Target hit | 2026-02-16 15:20:00 | 356.45 | 354.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 341.40 | 342.99 | 0.00 | ORB-short ORB[342.40,346.90] vol=2.4x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:55:00 | 340.42 | 342.58 | 0.00 | T1 1.5R @ 340.42 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 341.40 | 342.54 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 336.75 | 339.21 | 0.00 | ORB-short ORB[337.50,341.40] vol=3.8x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-02-25 11:50:00 | 337.52 | 338.63 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 329.65 | 333.17 | 0.00 | ORB-short ORB[333.15,337.00] vol=1.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-02-26 10:50:00 | 330.78 | 332.30 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 11:00:00 | 334.25 | 333.47 | 0.00 | ORB-long ORB[331.60,334.00] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 333.64 | 333.49 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:00:00 | 324.45 | 323.74 | 0.00 | ORB-long ORB[320.50,323.95] vol=1.7x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:10:00 | 325.61 | 324.28 | 0.00 | T1 1.5R @ 325.61 |
| Target hit | 2026-03-05 10:40:00 | 324.65 | 326.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 321.30 | 322.48 | 0.00 | ORB-short ORB[321.65,324.00] vol=1.9x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-03-06 11:20:00 | 321.98 | 322.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 319.00 | 316.89 | 0.00 | ORB-long ORB[314.00,318.10] vol=1.7x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:50:00 | 320.50 | 317.59 | 0.00 | T1 1.5R @ 320.50 |
| Target hit | 2026-03-10 13:35:00 | 320.00 | 320.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-03-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:00:00 | 315.20 | 315.60 | 0.00 | ORB-short ORB[315.35,317.65] vol=4.6x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:20:00 | 313.89 | 315.33 | 0.00 | T1 1.5R @ 313.89 |
| Target hit | 2026-03-13 15:20:00 | 313.60 | 313.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:40:00 | 315.10 | 312.73 | 0.00 | ORB-long ORB[309.65,312.90] vol=1.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-03-18 10:05:00 | 314.25 | 313.48 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:05:00 | 313.05 | 314.75 | 0.00 | ORB-short ORB[313.90,317.00] vol=3.2x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:50:00 | 311.65 | 313.87 | 0.00 | T1 1.5R @ 311.65 |
| Stop hit — per-position SL triggered | 2026-03-20 13:25:00 | 313.05 | 312.72 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-03-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-25 10:45:00 | 324.05 | 325.04 | 0.00 | ORB-short ORB[325.20,328.00] vol=1.6x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:00:00 | 322.96 | 324.64 | 0.00 | T1 1.5R @ 322.96 |
| Stop hit — per-position SL triggered | 2026-03-25 11:10:00 | 324.05 | 324.54 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:40:00 | 318.70 | 317.38 | 0.00 | ORB-long ORB[314.55,318.00] vol=3.0x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:45:00 | 320.46 | 318.11 | 0.00 | T1 1.5R @ 320.46 |
| Target hit | 2026-04-08 12:45:00 | 320.90 | 321.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2026-04-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:55:00 | 328.00 | 327.21 | 0.00 | ORB-long ORB[322.85,327.60] vol=1.8x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:45:00 | 329.65 | 327.62 | 0.00 | T1 1.5R @ 329.65 |
| Stop hit — per-position SL triggered | 2026-04-09 13:15:00 | 328.00 | 328.05 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 337.90 | 335.85 | 0.00 | ORB-long ORB[332.50,336.95] vol=1.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-04-17 11:05:00 | 337.08 | 335.87 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 11:00:00 | 334.80 | 336.91 | 0.00 | ORB-short ORB[337.35,341.55] vol=2.0x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 12:05:00 | 333.31 | 336.36 | 0.00 | T1 1.5R @ 333.31 |
| Stop hit — per-position SL triggered | 2026-04-27 14:45:00 | 334.80 | 335.00 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:55:00 | 329.60 | 333.34 | 0.00 | ORB-short ORB[333.15,336.70] vol=1.8x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 12:00:00 | 328.23 | 332.37 | 0.00 | T1 1.5R @ 328.23 |
| Target hit | 2026-05-04 15:20:00 | 327.90 | 330.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 322.50 | 323.48 | 0.00 | ORB-short ORB[322.65,325.75] vol=1.9x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-05-08 11:20:00 | 323.07 | 323.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 351.80 | 2026-02-09 15:20:00 | 351.25 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-02-12 11:10:00 | 351.65 | 2026-02-12 11:30:00 | 352.62 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-16 11:15:00 | 353.05 | 2026-02-16 12:15:00 | 354.18 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-16 11:15:00 | 353.05 | 2026-02-16 15:20:00 | 356.45 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-02-23 10:40:00 | 341.40 | 2026-02-23 10:55:00 | 340.42 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-23 10:40:00 | 341.40 | 2026-02-23 11:00:00 | 341.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 11:05:00 | 336.75 | 2026-02-25 11:50:00 | 337.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-26 10:15:00 | 329.65 | 2026-02-26 10:50:00 | 330.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-27 11:00:00 | 334.25 | 2026-02-27 11:05:00 | 333.64 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-03-05 10:00:00 | 324.45 | 2026-03-05 10:10:00 | 325.61 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-03-05 10:00:00 | 324.45 | 2026-03-05 10:40:00 | 324.65 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-03-06 10:45:00 | 321.30 | 2026-03-06 11:20:00 | 321.98 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-03-10 10:40:00 | 319.00 | 2026-03-10 10:50:00 | 320.50 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-10 10:40:00 | 319.00 | 2026-03-10 13:35:00 | 320.00 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-03-13 11:00:00 | 315.20 | 2026-03-13 12:20:00 | 313.89 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-13 11:00:00 | 315.20 | 2026-03-13 15:20:00 | 313.60 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-18 09:40:00 | 315.10 | 2026-03-18 10:05:00 | 314.25 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-20 11:05:00 | 313.05 | 2026-03-20 11:50:00 | 311.65 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-20 11:05:00 | 313.05 | 2026-03-20 13:25:00 | 313.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-25 10:45:00 | 324.05 | 2026-03-25 11:00:00 | 322.96 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-25 10:45:00 | 324.05 | 2026-03-25 11:10:00 | 324.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 09:40:00 | 318.70 | 2026-04-08 09:45:00 | 320.46 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-08 09:40:00 | 318.70 | 2026-04-08 12:45:00 | 320.90 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-09 10:55:00 | 328.00 | 2026-04-09 11:45:00 | 329.65 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-09 10:55:00 | 328.00 | 2026-04-09 13:15:00 | 328.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 11:00:00 | 337.90 | 2026-04-17 11:05:00 | 337.08 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-27 11:00:00 | 334.80 | 2026-04-27 12:05:00 | 333.31 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-04-27 11:00:00 | 334.80 | 2026-04-27 14:45:00 | 334.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 10:55:00 | 329.60 | 2026-05-04 12:00:00 | 328.23 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-04 10:55:00 | 329.60 | 2026-05-04 15:20:00 | 327.90 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-08 10:30:00 | 322.50 | 2026-05-08 11:20:00 | 323.07 | STOP_HIT | 1.00 | -0.18% |
