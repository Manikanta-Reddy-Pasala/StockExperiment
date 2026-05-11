# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36860 bars)
- **Last close:** 866.00
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
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 5
- **Avg / median % per leg:** -0.06% / 0.00%
- **Sum % (uncompounded):** -1.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.07% | 0.7% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.07% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.23% | -1.9% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.23% | -1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 7 | 36.8% | 2 | 12 | 5 | -0.06% | -1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:50:00 | 288.00 | 286.43 | 0.00 | ORB-long ORB[282.95,285.00] vol=2.8x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-05-18 11:30:00 | 286.27 | 286.87 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:55:00 | 355.45 | 357.80 | 0.00 | ORB-short ORB[356.00,360.00] vol=1.8x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-06-21 10:00:00 | 357.87 | 357.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 365.00 | 360.91 | 0.00 | ORB-long ORB[355.00,360.35] vol=3.2x ATR=2.33 |
| Stop hit — per-position SL triggered | 2024-07-01 09:35:00 | 362.67 | 361.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 11:10:00 | 295.00 | 291.50 | 0.00 | ORB-long ORB[288.05,292.00] vol=3.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:35:00 | 295.85 | 291.91 | 0.00 | T1 1.5R @ 295.85 |
| Stop hit — per-position SL triggered | 2024-08-07 11:45:00 | 295.00 | 292.15 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 375.00 | 377.57 | 0.00 | ORB-short ORB[377.00,381.00] vol=3.5x ATR=2.11 |
| Stop hit — per-position SL triggered | 2024-08-30 09:35:00 | 377.11 | 377.60 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:35:00 | 390.00 | 387.10 | 0.00 | ORB-long ORB[383.20,388.80] vol=3.0x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-09-24 09:40:00 | 387.61 | 387.13 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-10-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:25:00 | 356.50 | 354.12 | 0.00 | ORB-long ORB[351.00,356.00] vol=2.1x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 10:40:00 | 360.48 | 356.72 | 0.00 | T1 1.5R @ 360.48 |
| Target hit | 2024-10-09 13:30:00 | 358.05 | 358.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2024-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:40:00 | 332.35 | 337.09 | 0.00 | ORB-short ORB[341.00,344.65] vol=2.5x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-10-23 09:45:00 | 336.40 | 337.10 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-10-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:40:00 | 347.00 | 343.96 | 0.00 | ORB-long ORB[340.00,345.00] vol=1.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-10-24 10:50:00 | 345.10 | 345.00 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-11-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:10:00 | 334.90 | 331.05 | 0.00 | ORB-long ORB[325.00,329.50] vol=11.8x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-11-28 10:20:00 | 333.34 | 331.35 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-12-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:55:00 | 374.65 | 375.93 | 0.00 | ORB-short ORB[379.25,384.65] vol=11.5x ATR=3.31 |
| Stop hit — per-position SL triggered | 2024-12-05 10:00:00 | 377.96 | 376.10 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:25:00 | 321.35 | 320.78 | 0.00 | ORB-long ORB[315.35,319.45] vol=2.2x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:30:00 | 322.81 | 322.71 | 0.00 | T1 1.5R @ 322.81 |
| Target hit | 2025-02-20 13:25:00 | 325.55 | 325.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2025-03-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:05:00 | 313.00 | 315.62 | 0.00 | ORB-short ORB[314.70,319.00] vol=3.1x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:35:00 | 310.71 | 314.05 | 0.00 | T1 1.5R @ 310.71 |
| Stop hit — per-position SL triggered | 2025-03-12 12:20:00 | 313.00 | 313.18 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 449.65 | 454.39 | 0.00 | ORB-short ORB[453.30,458.40] vol=1.5x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:50:00 | 446.30 | 450.78 | 0.00 | T1 1.5R @ 446.30 |
| Stop hit — per-position SL triggered | 2025-04-29 10:35:00 | 449.65 | 449.53 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-18 09:50:00 | 288.00 | 2024-05-18 11:30:00 | 286.27 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-06-21 09:55:00 | 355.45 | 2024-06-21 10:00:00 | 357.87 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2024-07-01 09:30:00 | 365.00 | 2024-07-01 09:35:00 | 362.67 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2024-08-07 11:10:00 | 295.00 | 2024-08-07 11:35:00 | 295.85 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-08-07 11:10:00 | 295.00 | 2024-08-07 11:45:00 | 295.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 09:30:00 | 375.00 | 2024-08-30 09:35:00 | 377.11 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-09-24 09:35:00 | 390.00 | 2024-09-24 09:40:00 | 387.61 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-10-09 10:25:00 | 356.50 | 2024-10-09 10:40:00 | 360.48 | PARTIAL | 0.50 | 1.12% |
| BUY | retest1 | 2024-10-09 10:25:00 | 356.50 | 2024-10-09 13:30:00 | 358.05 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-23 09:40:00 | 332.35 | 2024-10-23 09:45:00 | 336.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2024-10-24 10:40:00 | 347.00 | 2024-10-24 10:50:00 | 345.10 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-11-28 10:10:00 | 334.90 | 2024-11-28 10:20:00 | 333.34 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-05 09:55:00 | 374.65 | 2024-12-05 10:00:00 | 377.96 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest1 | 2025-02-20 10:25:00 | 321.35 | 2025-02-20 11:30:00 | 322.81 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-02-20 10:25:00 | 321.35 | 2025-02-20 13:25:00 | 325.55 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2025-03-12 10:05:00 | 313.00 | 2025-03-12 11:35:00 | 310.71 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-03-12 10:05:00 | 313.00 | 2025-03-12 12:20:00 | 313.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-29 09:30:00 | 449.65 | 2025-04-29 09:50:00 | 446.30 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-04-29 09:30:00 | 449.65 | 2025-04-29 10:35:00 | 449.65 | STOP_HIT | 0.50 | 0.00% |
