# Aditya Birla Capital Ltd. (ABCAPITAL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 362.25
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 10
- **Target hits / Stop hits / Partials:** 3 / 10 / 4
- **Avg / median % per leg:** 0.14% / -0.24%
- **Sum % (uncompounded):** 2.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.27% | -2.5% |
| BUY @ 2nd Alert (retest1) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.27% | -2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.61% | 4.8% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.61% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 7 | 41.2% | 3 | 10 | 4 | 0.14% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:50:00 | 354.05 | 352.47 | 0.00 | ORB-long ORB[348.30,352.35] vol=1.6x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-02-09 11:50:00 | 352.55 | 352.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 344.40 | 342.46 | 0.00 | ORB-long ORB[340.15,343.80] vol=1.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:05:00 | 345.70 | 343.02 | 0.00 | T1 1.5R @ 345.70 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 344.40 | 343.11 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 344.90 | 346.92 | 0.00 | ORB-short ORB[348.75,350.95] vol=2.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 345.98 | 346.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:40:00 | 339.80 | 342.20 | 0.00 | ORB-short ORB[341.65,345.00] vol=1.8x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 341.17 | 342.09 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 358.40 | 356.28 | 0.00 | ORB-long ORB[351.50,355.00] vol=3.5x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-02-26 09:40:00 | 356.96 | 356.71 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:15:00 | 333.50 | 335.65 | 0.00 | ORB-short ORB[333.55,338.20] vol=2.0x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:40:00 | 332.07 | 335.02 | 0.00 | T1 1.5R @ 332.07 |
| Target hit | 2026-03-11 15:20:00 | 323.65 | 330.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 321.20 | 316.75 | 0.00 | ORB-long ORB[313.60,317.90] vol=2.0x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 319.76 | 317.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:45:00 | 315.20 | 316.21 | 0.00 | ORB-short ORB[315.25,317.90] vol=1.5x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:15:00 | 313.18 | 315.55 | 0.00 | T1 1.5R @ 313.18 |
| Target hit | 2026-03-20 11:15:00 | 314.40 | 314.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — SELL (started 2026-04-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:10:00 | 340.80 | 342.72 | 0.00 | ORB-short ORB[343.75,347.50] vol=2.7x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:20:00 | 338.90 | 342.23 | 0.00 | T1 1.5R @ 338.90 |
| Target hit | 2026-04-16 15:20:00 | 338.30 | 339.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 345.80 | 343.69 | 0.00 | ORB-long ORB[340.50,344.75] vol=1.9x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-04-21 11:25:00 | 344.96 | 344.03 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 353.40 | 351.67 | 0.00 | ORB-long ORB[346.95,351.60] vol=2.1x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 352.03 | 351.83 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 352.70 | 345.23 | 0.00 | ORB-long ORB[337.55,341.45] vol=2.2x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-04-29 10:35:00 | 350.78 | 347.59 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 351.20 | 347.75 | 0.00 | ORB-long ORB[344.90,348.45] vol=2.1x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-05-04 09:40:00 | 349.83 | 347.94 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:50:00 | 354.05 | 2026-02-09 11:50:00 | 352.55 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-12 10:50:00 | 344.40 | 2026-02-12 11:05:00 | 345.70 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-12 10:50:00 | 344.40 | 2026-02-12 11:15:00 | 344.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:40:00 | 344.90 | 2026-02-19 11:00:00 | 345.98 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-24 09:40:00 | 339.80 | 2026-02-24 09:45:00 | 341.17 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-26 09:30:00 | 358.40 | 2026-02-26 09:40:00 | 356.96 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-11 11:15:00 | 333.50 | 2026-03-11 11:40:00 | 332.07 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-03-11 11:15:00 | 333.50 | 2026-03-11 15:20:00 | 323.65 | TARGET_HIT | 0.50 | 2.95% |
| BUY | retest1 | 2026-03-17 10:15:00 | 321.20 | 2026-03-17 10:30:00 | 319.76 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-20 09:45:00 | 315.20 | 2026-03-20 10:15:00 | 313.18 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-03-20 09:45:00 | 315.20 | 2026-03-20 11:15:00 | 314.40 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2026-04-16 10:10:00 | 340.80 | 2026-04-16 10:20:00 | 338.90 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-16 10:10:00 | 340.80 | 2026-04-16 15:20:00 | 338.30 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2026-04-21 11:10:00 | 345.80 | 2026-04-21 11:25:00 | 344.96 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-23 10:00:00 | 353.40 | 2026-04-23 10:15:00 | 352.03 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-29 10:10:00 | 352.70 | 2026-04-29 10:35:00 | 350.78 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-05-04 09:35:00 | 351.20 | 2026-05-04 09:40:00 | 349.83 | STOP_HIT | 1.00 | -0.39% |
