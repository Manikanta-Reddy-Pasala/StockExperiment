# Balrampur Chini Mills Ltd. (BALRAMCHIN)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53844 bars)
- **Last close:** 522.00
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
| ENTRY1 | 106 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 19 |
| STOP_HIT | 87 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 148 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 87
- **Target hits / Stop hits / Partials:** 19 / 87 / 42
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 9.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 81 | 38 | 46.9% | 13 | 43 | 25 | 0.12% | 9.9% |
| BUY @ 2nd Alert (retest1) | 81 | 38 | 46.9% | 13 | 43 | 25 | 0.12% | 9.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 67 | 23 | 34.3% | 6 | 44 | 17 | -0.01% | -0.6% |
| SELL @ 2nd Alert (retest1) | 67 | 23 | 34.3% | 6 | 44 | 17 | -0.01% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 148 | 61 | 41.2% | 19 | 87 | 42 | 0.06% | 9.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:55:00 | 390.80 | 387.36 | 0.00 | ORB-long ORB[385.40,390.00] vol=2.8x ATR=1.07 |
| Stop hit — per-position SL triggered | 2023-05-22 11:00:00 | 389.73 | 387.45 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 10:55:00 | 389.70 | 391.50 | 0.00 | ORB-short ORB[389.80,394.45] vol=2.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2023-05-23 11:35:00 | 390.80 | 391.22 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 10:30:00 | 386.85 | 388.38 | 0.00 | ORB-short ORB[387.10,391.05] vol=1.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2023-05-25 10:45:00 | 387.87 | 388.32 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 11:00:00 | 389.75 | 391.88 | 0.00 | ORB-short ORB[390.45,392.90] vol=1.7x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 12:20:00 | 388.20 | 391.15 | 0.00 | T1 1.5R @ 388.20 |
| Stop hit — per-position SL triggered | 2023-05-26 14:30:00 | 389.75 | 390.42 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:35:00 | 393.50 | 391.89 | 0.00 | ORB-long ORB[389.00,393.05] vol=1.8x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 09:45:00 | 395.19 | 393.09 | 0.00 | T1 1.5R @ 395.19 |
| Stop hit — per-position SL triggered | 2023-05-29 09:55:00 | 393.50 | 393.12 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 09:40:00 | 393.80 | 392.08 | 0.00 | ORB-long ORB[390.40,391.90] vol=2.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2023-05-31 09:45:00 | 392.83 | 392.20 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 09:30:00 | 396.05 | 394.47 | 0.00 | ORB-long ORB[391.45,394.95] vol=3.2x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-06-01 09:35:00 | 395.18 | 394.62 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:45:00 | 393.45 | 392.20 | 0.00 | ORB-long ORB[390.25,393.10] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-06-02 09:55:00 | 392.58 | 392.29 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 09:30:00 | 391.40 | 392.99 | 0.00 | ORB-short ORB[391.70,395.70] vol=3.6x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 09:40:00 | 389.88 | 392.04 | 0.00 | T1 1.5R @ 389.88 |
| Stop hit — per-position SL triggered | 2023-06-05 09:45:00 | 391.40 | 392.01 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 09:55:00 | 387.30 | 389.61 | 0.00 | ORB-short ORB[389.25,391.85] vol=4.2x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-06-06 10:25:00 | 388.11 | 388.90 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 09:55:00 | 398.00 | 394.61 | 0.00 | ORB-long ORB[391.10,395.00] vol=5.1x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 10:05:00 | 399.87 | 396.43 | 0.00 | T1 1.5R @ 399.87 |
| Target hit | 2023-06-07 15:20:00 | 402.75 | 400.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2023-06-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 10:40:00 | 406.20 | 403.73 | 0.00 | ORB-long ORB[402.20,405.60] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2023-06-08 10:50:00 | 405.05 | 404.07 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:05:00 | 398.50 | 401.10 | 0.00 | ORB-short ORB[401.10,403.90] vol=3.0x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 11:15:00 | 397.03 | 400.81 | 0.00 | T1 1.5R @ 397.03 |
| Stop hit — per-position SL triggered | 2023-06-14 11:25:00 | 398.50 | 400.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:30:00 | 395.30 | 394.10 | 0.00 | ORB-long ORB[392.00,394.90] vol=2.5x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 09:45:00 | 397.33 | 395.46 | 0.00 | T1 1.5R @ 397.33 |
| Target hit | 2023-06-16 10:15:00 | 396.60 | 396.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 388.55 | 390.96 | 0.00 | ORB-short ORB[390.00,395.40] vol=4.0x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-06-20 09:35:00 | 390.01 | 390.80 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-06-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:20:00 | 401.75 | 398.85 | 0.00 | ORB-long ORB[396.35,401.20] vol=2.5x ATR=1.57 |
| Stop hit — per-position SL triggered | 2023-06-21 10:45:00 | 400.18 | 399.53 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 10:35:00 | 383.65 | 386.30 | 0.00 | ORB-short ORB[385.00,388.40] vol=2.1x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-06-27 10:55:00 | 384.79 | 385.75 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-06-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 09:35:00 | 384.15 | 382.73 | 0.00 | ORB-long ORB[381.35,383.95] vol=1.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-06-28 10:05:00 | 383.14 | 383.33 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:30:00 | 384.80 | 384.19 | 0.00 | ORB-long ORB[382.45,384.70] vol=2.1x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-06-30 09:35:00 | 383.72 | 384.24 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 09:35:00 | 384.40 | 386.01 | 0.00 | ORB-short ORB[384.80,387.80] vol=1.7x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-07-03 09:40:00 | 385.40 | 385.88 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:55:00 | 382.30 | 384.06 | 0.00 | ORB-short ORB[383.00,385.50] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-07-04 10:05:00 | 383.15 | 383.90 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:50:00 | 387.20 | 385.23 | 0.00 | ORB-long ORB[382.45,385.50] vol=2.5x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-07-05 10:20:00 | 386.24 | 385.71 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:45:00 | 382.45 | 385.28 | 0.00 | ORB-short ORB[382.55,386.80] vol=1.7x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-07-07 11:05:00 | 383.43 | 384.99 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:25:00 | 384.65 | 382.30 | 0.00 | ORB-long ORB[380.30,384.05] vol=2.3x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 10:45:00 | 386.10 | 382.91 | 0.00 | T1 1.5R @ 386.10 |
| Stop hit — per-position SL triggered | 2023-07-10 11:00:00 | 384.65 | 383.18 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 09:35:00 | 380.95 | 382.24 | 0.00 | ORB-short ORB[381.50,384.10] vol=3.1x ATR=0.83 |
| Stop hit — per-position SL triggered | 2023-07-11 09:50:00 | 381.78 | 381.97 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:55:00 | 388.35 | 385.96 | 0.00 | ORB-long ORB[382.00,386.45] vol=4.1x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:00:00 | 389.91 | 388.02 | 0.00 | T1 1.5R @ 389.91 |
| Stop hit — per-position SL triggered | 2023-07-12 10:20:00 | 388.35 | 388.71 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:15:00 | 379.25 | 377.23 | 0.00 | ORB-long ORB[374.25,377.70] vol=2.2x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-07-14 10:30:00 | 377.89 | 377.47 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:30:00 | 384.35 | 381.59 | 0.00 | ORB-long ORB[379.50,382.00] vol=2.3x ATR=1.23 |
| Stop hit — per-position SL triggered | 2023-07-17 10:20:00 | 383.12 | 383.24 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:40:00 | 383.40 | 382.11 | 0.00 | ORB-long ORB[380.50,382.50] vol=2.4x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-07-19 09:55:00 | 382.50 | 382.64 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-07-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 09:30:00 | 388.25 | 385.37 | 0.00 | ORB-long ORB[382.00,387.00] vol=4.6x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 09:35:00 | 389.94 | 387.08 | 0.00 | T1 1.5R @ 389.94 |
| Stop hit — per-position SL triggered | 2023-07-20 09:45:00 | 388.25 | 387.56 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-07-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-21 09:40:00 | 384.15 | 385.66 | 0.00 | ORB-short ORB[384.40,388.50] vol=2.5x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-07-21 09:45:00 | 385.46 | 385.51 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:30:00 | 397.25 | 395.56 | 0.00 | ORB-long ORB[393.05,396.10] vol=2.0x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-07-24 09:40:00 | 395.58 | 396.06 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-07-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:25:00 | 403.40 | 400.41 | 0.00 | ORB-long ORB[397.00,401.20] vol=2.9x ATR=1.35 |
| Stop hit — per-position SL triggered | 2023-07-26 10:30:00 | 402.05 | 400.51 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-08-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:35:00 | 403.30 | 404.33 | 0.00 | ORB-short ORB[404.00,406.40] vol=1.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-08-01 10:50:00 | 404.52 | 404.30 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-08-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:50:00 | 391.70 | 394.68 | 0.00 | ORB-short ORB[394.05,397.35] vol=1.5x ATR=1.28 |
| Stop hit — per-position SL triggered | 2023-08-09 11:40:00 | 392.98 | 393.95 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-08-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:30:00 | 398.70 | 396.91 | 0.00 | ORB-long ORB[393.75,397.60] vol=2.4x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-08-10 09:35:00 | 397.61 | 397.10 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-08-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:30:00 | 394.00 | 395.29 | 0.00 | ORB-short ORB[394.95,396.85] vol=1.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2023-08-11 09:50:00 | 395.25 | 394.57 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-08-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 09:30:00 | 389.75 | 391.11 | 0.00 | ORB-short ORB[390.00,392.80] vol=1.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2023-08-24 09:45:00 | 390.80 | 390.59 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:15:00 | 389.20 | 391.74 | 0.00 | ORB-short ORB[391.10,394.00] vol=1.5x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 10:30:00 | 387.49 | 391.11 | 0.00 | T1 1.5R @ 387.49 |
| Stop hit — per-position SL triggered | 2023-08-25 10:40:00 | 389.20 | 390.86 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-08-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:45:00 | 393.70 | 391.81 | 0.00 | ORB-long ORB[389.70,392.60] vol=1.5x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 10:25:00 | 395.39 | 393.06 | 0.00 | T1 1.5R @ 395.39 |
| Target hit | 2023-08-30 12:50:00 | 395.50 | 396.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — BUY (started 2023-09-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:20:00 | 397.00 | 394.09 | 0.00 | ORB-long ORB[390.30,392.80] vol=1.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-09-01 10:35:00 | 395.52 | 394.37 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:15:00 | 414.25 | 411.30 | 0.00 | ORB-long ORB[406.50,412.00] vol=2.3x ATR=1.21 |
| Stop hit — per-position SL triggered | 2023-09-06 10:25:00 | 413.04 | 411.68 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:05:00 | 410.55 | 411.99 | 0.00 | ORB-short ORB[412.10,417.70] vol=1.6x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 10:20:00 | 408.94 | 411.69 | 0.00 | T1 1.5R @ 408.94 |
| Stop hit — per-position SL triggered | 2023-09-08 10:50:00 | 410.55 | 411.30 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:30:00 | 416.00 | 419.79 | 0.00 | ORB-short ORB[419.00,423.05] vol=1.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2023-09-22 09:35:00 | 418.39 | 419.47 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-09-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:35:00 | 451.10 | 450.40 | 0.00 | ORB-long ORB[444.10,450.90] vol=2.3x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 10:00:00 | 454.32 | 451.22 | 0.00 | T1 1.5R @ 454.32 |
| Target hit | 2023-09-28 10:45:00 | 453.50 | 455.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2023-10-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 09:40:00 | 433.75 | 430.15 | 0.00 | ORB-long ORB[427.10,431.60] vol=3.5x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 09:50:00 | 436.82 | 431.78 | 0.00 | T1 1.5R @ 436.82 |
| Stop hit — per-position SL triggered | 2023-10-04 10:00:00 | 433.75 | 432.15 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 09:50:00 | 426.95 | 429.36 | 0.00 | ORB-short ORB[427.65,432.90] vol=1.8x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 10:25:00 | 424.40 | 428.43 | 0.00 | T1 1.5R @ 424.40 |
| Target hit | 2023-10-05 15:20:00 | 424.80 | 426.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2023-10-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 10:10:00 | 429.60 | 427.73 | 0.00 | ORB-long ORB[425.05,429.10] vol=2.3x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 10:15:00 | 431.37 | 428.78 | 0.00 | T1 1.5R @ 431.37 |
| Target hit | 2023-10-06 11:05:00 | 430.15 | 430.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — SELL (started 2023-10-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 10:05:00 | 425.50 | 427.38 | 0.00 | ORB-short ORB[426.10,429.10] vol=1.7x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-10-11 10:15:00 | 426.50 | 427.05 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 11:15:00 | 420.80 | 418.81 | 0.00 | ORB-long ORB[415.10,419.60] vol=5.2x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-10-13 11:20:00 | 419.82 | 418.88 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:45:00 | 425.50 | 423.37 | 0.00 | ORB-long ORB[421.55,424.00] vol=2.9x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 09:55:00 | 427.04 | 425.43 | 0.00 | T1 1.5R @ 427.04 |
| Target hit | 2023-10-18 10:10:00 | 426.20 | 426.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — SELL (started 2023-10-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:35:00 | 418.80 | 420.34 | 0.00 | ORB-short ORB[419.10,423.50] vol=1.8x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-10-19 09:40:00 | 420.23 | 420.37 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-10-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 09:45:00 | 425.20 | 423.90 | 0.00 | ORB-long ORB[421.05,425.00] vol=3.4x ATR=1.54 |
| Stop hit — per-position SL triggered | 2023-10-20 09:55:00 | 423.66 | 423.88 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:50:00 | 419.70 | 417.16 | 0.00 | ORB-long ORB[413.70,417.85] vol=3.9x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 12:25:00 | 421.32 | 418.71 | 0.00 | T1 1.5R @ 421.32 |
| Target hit | 2023-11-02 15:20:00 | 426.50 | 421.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2023-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 11:00:00 | 427.05 | 425.79 | 0.00 | ORB-long ORB[423.80,427.00] vol=3.4x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 11:05:00 | 428.49 | 426.80 | 0.00 | T1 1.5R @ 428.49 |
| Stop hit — per-position SL triggered | 2023-11-06 11:40:00 | 427.05 | 427.41 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-12-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:05:00 | 396.50 | 398.56 | 0.00 | ORB-short ORB[398.30,401.20] vol=1.6x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 10:15:00 | 394.91 | 398.05 | 0.00 | T1 1.5R @ 394.91 |
| Stop hit — per-position SL triggered | 2023-12-12 11:20:00 | 396.50 | 397.41 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 11:00:00 | 380.00 | 382.88 | 0.00 | ORB-short ORB[381.15,385.95] vol=3.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2023-12-14 11:15:00 | 381.27 | 382.68 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-12-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:05:00 | 394.75 | 396.99 | 0.00 | ORB-short ORB[396.55,399.10] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-12-20 10:20:00 | 395.56 | 396.42 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:55:00 | 389.40 | 388.42 | 0.00 | ORB-long ORB[387.00,388.95] vol=1.6x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:00:00 | 390.91 | 388.79 | 0.00 | T1 1.5R @ 390.91 |
| Stop hit — per-position SL triggered | 2023-12-22 10:35:00 | 389.40 | 389.28 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:15:00 | 392.60 | 391.21 | 0.00 | ORB-long ORB[389.05,391.45] vol=2.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-12-27 10:30:00 | 391.70 | 391.35 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:35:00 | 407.90 | 406.59 | 0.00 | ORB-long ORB[403.05,407.40] vol=4.3x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-01-03 09:40:00 | 406.91 | 406.55 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 09:30:00 | 405.85 | 407.40 | 0.00 | ORB-short ORB[406.35,409.85] vol=3.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-01-05 09:40:00 | 406.73 | 406.96 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:40:00 | 399.05 | 401.59 | 0.00 | ORB-short ORB[402.15,406.30] vol=1.9x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 14:35:00 | 396.90 | 399.09 | 0.00 | T1 1.5R @ 396.90 |
| Target hit | 2024-01-08 15:20:00 | 395.50 | 398.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2024-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 10:45:00 | 396.00 | 397.44 | 0.00 | ORB-short ORB[397.95,401.50] vol=2.4x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 10:55:00 | 394.39 | 397.12 | 0.00 | T1 1.5R @ 394.39 |
| Stop hit — per-position SL triggered | 2024-01-09 11:15:00 | 396.00 | 397.04 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:50:00 | 386.95 | 389.44 | 0.00 | ORB-short ORB[389.50,392.85] vol=2.0x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-01-11 11:00:00 | 387.87 | 389.33 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:55:00 | 390.00 | 392.04 | 0.00 | ORB-short ORB[391.80,394.00] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-01-15 10:00:00 | 391.11 | 391.97 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-01-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-16 09:50:00 | 389.90 | 390.73 | 0.00 | ORB-short ORB[390.00,392.35] vol=3.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-01-16 10:05:00 | 390.96 | 390.71 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-01-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:50:00 | 378.80 | 383.50 | 0.00 | ORB-short ORB[384.20,388.65] vol=3.7x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-01-18 09:55:00 | 380.36 | 383.10 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 11:00:00 | 389.50 | 387.93 | 0.00 | ORB-long ORB[385.30,388.65] vol=4.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-01-19 11:25:00 | 388.23 | 388.07 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:55:00 | 381.95 | 384.19 | 0.00 | ORB-short ORB[383.00,385.60] vol=2.1x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-01-23 10:10:00 | 383.09 | 383.39 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-01-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 09:50:00 | 380.95 | 379.50 | 0.00 | ORB-long ORB[377.70,380.50] vol=1.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 11:45:00 | 383.25 | 380.37 | 0.00 | T1 1.5R @ 383.25 |
| Stop hit — per-position SL triggered | 2024-01-24 12:50:00 | 380.95 | 381.84 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 09:30:00 | 390.75 | 388.81 | 0.00 | ORB-long ORB[384.75,389.60] vol=4.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 09:50:00 | 393.39 | 390.61 | 0.00 | T1 1.5R @ 393.39 |
| Stop hit — per-position SL triggered | 2024-01-25 10:00:00 | 390.75 | 390.67 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 11:10:00 | 386.50 | 387.41 | 0.00 | ORB-short ORB[387.75,390.90] vol=1.8x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-01-29 11:45:00 | 387.46 | 387.34 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:25:00 | 398.05 | 394.06 | 0.00 | ORB-long ORB[390.25,394.30] vol=7.1x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-01-30 10:40:00 | 396.26 | 394.51 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-02-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:00:00 | 393.60 | 394.84 | 0.00 | ORB-short ORB[394.00,396.90] vol=1.7x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 10:35:00 | 392.06 | 394.41 | 0.00 | T1 1.5R @ 392.06 |
| Stop hit — per-position SL triggered | 2024-02-01 11:00:00 | 393.60 | 394.25 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-02-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:35:00 | 393.35 | 391.84 | 0.00 | ORB-long ORB[390.20,392.50] vol=1.6x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 10:00:00 | 394.96 | 393.35 | 0.00 | T1 1.5R @ 394.96 |
| Target hit | 2024-02-02 11:45:00 | 395.20 | 395.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 77 — BUY (started 2024-02-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 10:20:00 | 402.55 | 400.01 | 0.00 | ORB-long ORB[397.05,400.00] vol=5.0x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 10:50:00 | 405.01 | 402.11 | 0.00 | T1 1.5R @ 405.01 |
| Stop hit — per-position SL triggered | 2024-02-07 11:30:00 | 402.55 | 403.05 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 09:45:00 | 380.35 | 383.17 | 0.00 | ORB-short ORB[382.60,384.95] vol=1.5x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-02-12 09:50:00 | 381.47 | 382.87 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-02-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 10:20:00 | 376.80 | 374.65 | 0.00 | ORB-long ORB[372.60,375.65] vol=2.7x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 10:25:00 | 378.44 | 375.28 | 0.00 | T1 1.5R @ 378.44 |
| Stop hit — per-position SL triggered | 2024-02-15 11:10:00 | 376.80 | 376.12 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-02-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 11:00:00 | 381.25 | 382.27 | 0.00 | ORB-short ORB[381.80,385.90] vol=2.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-02-21 12:00:00 | 382.36 | 382.04 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:30:00 | 373.05 | 375.35 | 0.00 | ORB-short ORB[374.45,378.20] vol=2.4x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-02-22 09:40:00 | 374.75 | 375.25 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:40:00 | 378.50 | 379.85 | 0.00 | ORB-short ORB[378.55,381.90] vol=2.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-02-26 09:45:00 | 379.84 | 379.80 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 11:15:00 | 379.10 | 377.65 | 0.00 | ORB-long ORB[376.10,379.00] vol=5.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 11:55:00 | 380.52 | 378.59 | 0.00 | T1 1.5R @ 380.52 |
| Target hit | 2024-02-27 15:20:00 | 383.00 | 381.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — SELL (started 2024-02-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 11:10:00 | 375.60 | 380.59 | 0.00 | ORB-short ORB[382.00,385.40] vol=2.2x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 12:00:00 | 373.75 | 379.60 | 0.00 | T1 1.5R @ 373.75 |
| Stop hit — per-position SL triggered | 2024-02-28 14:00:00 | 375.60 | 377.53 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 09:45:00 | 378.00 | 376.51 | 0.00 | ORB-long ORB[374.30,377.35] vol=2.8x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-03-04 10:00:00 | 376.74 | 376.79 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:35:00 | 376.40 | 378.44 | 0.00 | ORB-short ORB[377.00,380.80] vol=2.1x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-03-06 09:40:00 | 377.72 | 378.36 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-03-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:10:00 | 377.35 | 376.13 | 0.00 | ORB-long ORB[374.00,376.35] vol=3.6x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 11:00:00 | 378.85 | 376.84 | 0.00 | T1 1.5R @ 378.85 |
| Target hit | 2024-03-07 12:25:00 | 377.55 | 377.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 88 — SELL (started 2024-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 11:10:00 | 375.25 | 375.69 | 0.00 | ORB-short ORB[375.50,379.45] vol=3.0x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:25:00 | 373.84 | 375.58 | 0.00 | T1 1.5R @ 373.84 |
| Target hit | 2024-03-11 12:15:00 | 375.00 | 374.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 89 — SELL (started 2024-03-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 09:50:00 | 355.00 | 356.91 | 0.00 | ORB-short ORB[356.05,359.75] vol=3.1x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:55:00 | 353.15 | 356.70 | 0.00 | T1 1.5R @ 353.15 |
| Target hit | 2024-03-15 11:20:00 | 354.75 | 353.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 90 — BUY (started 2024-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 11:05:00 | 360.70 | 358.53 | 0.00 | ORB-long ORB[357.05,359.55] vol=1.8x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 12:20:00 | 362.09 | 359.22 | 0.00 | T1 1.5R @ 362.09 |
| Target hit | 2024-03-19 15:20:00 | 361.95 | 361.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 91 — BUY (started 2024-03-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:20:00 | 364.30 | 362.13 | 0.00 | ORB-long ORB[360.15,362.20] vol=1.5x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-03-21 10:30:00 | 363.35 | 362.43 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-03-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 10:30:00 | 370.10 | 367.27 | 0.00 | ORB-long ORB[363.30,367.80] vol=2.1x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-03-22 10:40:00 | 368.95 | 367.49 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-03-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 09:50:00 | 364.55 | 366.78 | 0.00 | ORB-short ORB[366.00,369.10] vol=1.9x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-03-27 10:40:00 | 365.62 | 366.02 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 10:55:00 | 360.00 | 361.67 | 0.00 | ORB-short ORB[360.95,364.25] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-03-28 13:15:00 | 360.88 | 360.83 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 09:35:00 | 378.30 | 376.79 | 0.00 | ORB-long ORB[374.75,377.40] vol=1.6x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-04-02 09:45:00 | 377.27 | 376.88 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-04-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:50:00 | 388.35 | 384.44 | 0.00 | ORB-long ORB[380.50,383.50] vol=5.3x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-04-03 09:55:00 | 386.91 | 384.76 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 383.95 | 386.07 | 0.00 | ORB-short ORB[386.00,388.75] vol=2.1x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 11:00:00 | 382.31 | 385.03 | 0.00 | T1 1.5R @ 382.31 |
| Target hit | 2024-04-04 15:20:00 | 383.60 | 383.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 98 — BUY (started 2024-04-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 10:25:00 | 386.30 | 384.25 | 0.00 | ORB-long ORB[382.80,385.00] vol=1.5x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-05 10:40:00 | 387.96 | 385.41 | 0.00 | T1 1.5R @ 387.96 |
| Target hit | 2024-04-05 11:25:00 | 388.20 | 388.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 99 — BUY (started 2024-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 11:05:00 | 369.60 | 367.61 | 0.00 | ORB-long ORB[364.55,368.20] vol=1.8x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 11:50:00 | 371.15 | 368.42 | 0.00 | T1 1.5R @ 371.15 |
| Target hit | 2024-04-16 15:20:00 | 373.45 | 370.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 100 — BUY (started 2024-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 09:35:00 | 376.25 | 372.46 | 0.00 | ORB-long ORB[368.00,372.00] vol=5.4x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-04-22 09:40:00 | 374.85 | 372.87 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-04-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:50:00 | 390.80 | 389.12 | 0.00 | ORB-long ORB[387.50,389.95] vol=1.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-04-26 09:55:00 | 389.58 | 389.18 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-05-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:30:00 | 401.55 | 398.78 | 0.00 | ORB-long ORB[394.05,398.00] vol=5.6x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-05-03 09:35:00 | 400.26 | 399.07 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 388.75 | 391.12 | 0.00 | ORB-short ORB[390.75,393.90] vol=2.0x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:40:00 | 386.93 | 389.66 | 0.00 | T1 1.5R @ 386.93 |
| Stop hit — per-position SL triggered | 2024-05-06 10:30:00 | 388.75 | 388.72 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:35:00 | 381.10 | 383.98 | 0.00 | ORB-short ORB[383.65,387.55] vol=2.3x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:00:00 | 379.24 | 382.98 | 0.00 | T1 1.5R @ 379.24 |
| Target hit | 2024-05-07 13:40:00 | 380.40 | 380.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 105 — BUY (started 2024-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 11:05:00 | 378.60 | 377.80 | 0.00 | ORB-long ORB[374.00,378.30] vol=1.5x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 11:15:00 | 379.95 | 377.87 | 0.00 | T1 1.5R @ 379.95 |
| Stop hit — per-position SL triggered | 2024-05-08 14:25:00 | 378.60 | 378.90 | 0.00 | SL hit |

### Cycle 106 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:15:00 | 378.75 | 380.44 | 0.00 | ORB-short ORB[380.50,382.35] vol=2.1x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:40:00 | 377.41 | 379.84 | 0.00 | T1 1.5R @ 377.41 |
| Stop hit — per-position SL triggered | 2024-05-09 11:35:00 | 378.75 | 378.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-22 10:55:00 | 390.80 | 2023-05-22 11:00:00 | 389.73 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-05-23 10:55:00 | 389.70 | 2023-05-23 11:35:00 | 390.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-05-25 10:30:00 | 386.85 | 2023-05-25 10:45:00 | 387.87 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-05-26 11:00:00 | 389.75 | 2023-05-26 12:20:00 | 388.20 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-05-26 11:00:00 | 389.75 | 2023-05-26 14:30:00 | 389.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-29 09:35:00 | 393.50 | 2023-05-29 09:45:00 | 395.19 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-05-29 09:35:00 | 393.50 | 2023-05-29 09:55:00 | 393.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-31 09:40:00 | 393.80 | 2023-05-31 09:45:00 | 392.83 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-01 09:30:00 | 396.05 | 2023-06-01 09:35:00 | 395.18 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-02 09:45:00 | 393.45 | 2023-06-02 09:55:00 | 392.58 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-05 09:30:00 | 391.40 | 2023-06-05 09:40:00 | 389.88 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-06-05 09:30:00 | 391.40 | 2023-06-05 09:45:00 | 391.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-06 09:55:00 | 387.30 | 2023-06-06 10:25:00 | 388.11 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-07 09:55:00 | 398.00 | 2023-06-07 10:05:00 | 399.87 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-06-07 09:55:00 | 398.00 | 2023-06-07 15:20:00 | 402.75 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2023-06-08 10:40:00 | 406.20 | 2023-06-08 10:50:00 | 405.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-14 11:05:00 | 398.50 | 2023-06-14 11:15:00 | 397.03 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-06-14 11:05:00 | 398.50 | 2023-06-14 11:25:00 | 398.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 09:30:00 | 395.30 | 2023-06-16 09:45:00 | 397.33 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-06-16 09:30:00 | 395.30 | 2023-06-16 10:15:00 | 396.60 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2023-06-20 09:30:00 | 388.55 | 2023-06-20 09:35:00 | 390.01 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-06-21 10:20:00 | 401.75 | 2023-06-21 10:45:00 | 400.18 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-06-27 10:35:00 | 383.65 | 2023-06-27 10:55:00 | 384.79 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-06-28 09:35:00 | 384.15 | 2023-06-28 10:05:00 | 383.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-30 09:30:00 | 384.80 | 2023-06-30 09:35:00 | 383.72 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-03 09:35:00 | 384.40 | 2023-07-03 09:40:00 | 385.40 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-04 09:55:00 | 382.30 | 2023-07-04 10:05:00 | 383.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-05 09:50:00 | 387.20 | 2023-07-05 10:20:00 | 386.24 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-07 10:45:00 | 382.45 | 2023-07-07 11:05:00 | 383.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-10 10:25:00 | 384.65 | 2023-07-10 10:45:00 | 386.10 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-07-10 10:25:00 | 384.65 | 2023-07-10 11:00:00 | 384.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-11 09:35:00 | 380.95 | 2023-07-11 09:50:00 | 381.78 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-12 09:55:00 | 388.35 | 2023-07-12 10:00:00 | 389.91 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-07-12 09:55:00 | 388.35 | 2023-07-12 10:20:00 | 388.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 10:15:00 | 379.25 | 2023-07-14 10:30:00 | 377.89 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-07-17 09:30:00 | 384.35 | 2023-07-17 10:20:00 | 383.12 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-07-19 09:40:00 | 383.40 | 2023-07-19 09:55:00 | 382.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-20 09:30:00 | 388.25 | 2023-07-20 09:35:00 | 389.94 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-20 09:30:00 | 388.25 | 2023-07-20 09:45:00 | 388.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-21 09:40:00 | 384.15 | 2023-07-21 09:45:00 | 385.46 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-07-24 09:30:00 | 397.25 | 2023-07-24 09:40:00 | 395.58 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-07-26 10:25:00 | 403.40 | 2023-07-26 10:30:00 | 402.05 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-08-01 10:35:00 | 403.30 | 2023-08-01 10:50:00 | 404.52 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-08-09 10:50:00 | 391.70 | 2023-08-09 11:40:00 | 392.98 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-08-10 09:30:00 | 398.70 | 2023-08-10 09:35:00 | 397.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-11 09:30:00 | 394.00 | 2023-08-11 09:50:00 | 395.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-08-24 09:30:00 | 389.75 | 2023-08-24 09:45:00 | 390.80 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-25 10:15:00 | 389.20 | 2023-08-25 10:30:00 | 387.49 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-08-25 10:15:00 | 389.20 | 2023-08-25 10:40:00 | 389.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-30 09:45:00 | 393.70 | 2023-08-30 10:25:00 | 395.39 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-08-30 09:45:00 | 393.70 | 2023-08-30 12:50:00 | 395.50 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2023-09-01 10:20:00 | 397.00 | 2023-09-01 10:35:00 | 395.52 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-09-06 10:15:00 | 414.25 | 2023-09-06 10:25:00 | 413.04 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-09-08 10:05:00 | 410.55 | 2023-09-08 10:20:00 | 408.94 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-09-08 10:05:00 | 410.55 | 2023-09-08 10:50:00 | 410.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-22 09:30:00 | 416.00 | 2023-09-22 09:35:00 | 418.39 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2023-09-28 09:35:00 | 451.10 | 2023-09-28 10:00:00 | 454.32 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2023-09-28 09:35:00 | 451.10 | 2023-09-28 10:45:00 | 453.50 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2023-10-04 09:40:00 | 433.75 | 2023-10-04 09:50:00 | 436.82 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2023-10-04 09:40:00 | 433.75 | 2023-10-04 10:00:00 | 433.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-05 09:50:00 | 426.95 | 2023-10-05 10:25:00 | 424.40 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2023-10-05 09:50:00 | 426.95 | 2023-10-05 15:20:00 | 424.80 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2023-10-06 10:10:00 | 429.60 | 2023-10-06 10:15:00 | 431.37 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-10-06 10:10:00 | 429.60 | 2023-10-06 11:05:00 | 430.15 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2023-10-11 10:05:00 | 425.50 | 2023-10-11 10:15:00 | 426.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-10-13 11:15:00 | 420.80 | 2023-10-13 11:20:00 | 419.82 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-18 09:45:00 | 425.50 | 2023-10-18 09:55:00 | 427.04 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-10-18 09:45:00 | 425.50 | 2023-10-18 10:10:00 | 426.20 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2023-10-19 09:35:00 | 418.80 | 2023-10-19 09:40:00 | 420.23 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-10-20 09:45:00 | 425.20 | 2023-10-20 09:55:00 | 423.66 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-11-02 10:50:00 | 419.70 | 2023-11-02 12:25:00 | 421.32 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-11-02 10:50:00 | 419.70 | 2023-11-02 15:20:00 | 426.50 | TARGET_HIT | 0.50 | 1.62% |
| BUY | retest1 | 2023-11-06 11:00:00 | 427.05 | 2023-11-06 11:05:00 | 428.49 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-11-06 11:00:00 | 427.05 | 2023-11-06 11:40:00 | 427.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-12 10:05:00 | 396.50 | 2023-12-12 10:15:00 | 394.91 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-12-12 10:05:00 | 396.50 | 2023-12-12 11:20:00 | 396.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-14 11:00:00 | 380.00 | 2023-12-14 11:15:00 | 381.27 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-20 10:05:00 | 394.75 | 2023-12-20 10:20:00 | 395.56 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-22 09:55:00 | 389.40 | 2023-12-22 10:00:00 | 390.91 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-12-22 09:55:00 | 389.40 | 2023-12-22 10:35:00 | 389.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-27 10:15:00 | 392.60 | 2023-12-27 10:30:00 | 391.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-01-03 09:35:00 | 407.90 | 2024-01-03 09:40:00 | 406.91 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-05 09:30:00 | 405.85 | 2024-01-05 09:40:00 | 406.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-08 09:40:00 | 399.05 | 2024-01-08 14:35:00 | 396.90 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-01-08 09:40:00 | 399.05 | 2024-01-08 15:20:00 | 395.50 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2024-01-09 10:45:00 | 396.00 | 2024-01-09 10:55:00 | 394.39 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-01-09 10:45:00 | 396.00 | 2024-01-09 11:15:00 | 396.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-11 10:50:00 | 386.95 | 2024-01-11 11:00:00 | 387.87 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-15 09:55:00 | 390.00 | 2024-01-15 10:00:00 | 391.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-16 09:50:00 | 389.90 | 2024-01-16 10:05:00 | 390.96 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-18 09:50:00 | 378.80 | 2024-01-18 09:55:00 | 380.36 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-01-19 11:00:00 | 389.50 | 2024-01-19 11:25:00 | 388.23 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-01-23 09:55:00 | 381.95 | 2024-01-23 10:10:00 | 383.09 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-01-24 09:50:00 | 380.95 | 2024-01-24 11:45:00 | 383.25 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-01-24 09:50:00 | 380.95 | 2024-01-24 12:50:00 | 380.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-25 09:30:00 | 390.75 | 2024-01-25 09:50:00 | 393.39 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-01-25 09:30:00 | 390.75 | 2024-01-25 10:00:00 | 390.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-29 11:10:00 | 386.50 | 2024-01-29 11:45:00 | 387.46 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-01-30 10:25:00 | 398.05 | 2024-01-30 10:40:00 | 396.26 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-02-01 10:00:00 | 393.60 | 2024-02-01 10:35:00 | 392.06 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-01 10:00:00 | 393.60 | 2024-02-01 11:00:00 | 393.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 09:35:00 | 393.35 | 2024-02-02 10:00:00 | 394.96 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-02-02 09:35:00 | 393.35 | 2024-02-02 11:45:00 | 395.20 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2024-02-07 10:20:00 | 402.55 | 2024-02-07 10:50:00 | 405.01 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-02-07 10:20:00 | 402.55 | 2024-02-07 11:30:00 | 402.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-12 09:45:00 | 380.35 | 2024-02-12 09:50:00 | 381.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-02-15 10:20:00 | 376.80 | 2024-02-15 10:25:00 | 378.44 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-02-15 10:20:00 | 376.80 | 2024-02-15 11:10:00 | 376.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-21 11:00:00 | 381.25 | 2024-02-21 12:00:00 | 382.36 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-22 09:30:00 | 373.05 | 2024-02-22 09:40:00 | 374.75 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-02-26 09:40:00 | 378.50 | 2024-02-26 09:45:00 | 379.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-02-27 11:15:00 | 379.10 | 2024-02-27 11:55:00 | 380.52 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-02-27 11:15:00 | 379.10 | 2024-02-27 15:20:00 | 383.00 | TARGET_HIT | 0.50 | 1.03% |
| SELL | retest1 | 2024-02-28 11:10:00 | 375.60 | 2024-02-28 12:00:00 | 373.75 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-02-28 11:10:00 | 375.60 | 2024-02-28 14:00:00 | 375.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-04 09:45:00 | 378.00 | 2024-03-04 10:00:00 | 376.74 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-03-06 09:35:00 | 376.40 | 2024-03-06 09:40:00 | 377.72 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-03-07 10:10:00 | 377.35 | 2024-03-07 11:00:00 | 378.85 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-03-07 10:10:00 | 377.35 | 2024-03-07 12:25:00 | 377.55 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-03-11 11:10:00 | 375.25 | 2024-03-11 11:25:00 | 373.84 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-03-11 11:10:00 | 375.25 | 2024-03-11 12:15:00 | 375.00 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2024-03-15 09:50:00 | 355.00 | 2024-03-15 09:55:00 | 353.15 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-03-15 09:50:00 | 355.00 | 2024-03-15 11:20:00 | 354.75 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-03-19 11:05:00 | 360.70 | 2024-03-19 12:20:00 | 362.09 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-03-19 11:05:00 | 360.70 | 2024-03-19 15:20:00 | 361.95 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-03-21 10:20:00 | 364.30 | 2024-03-21 10:30:00 | 363.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-03-22 10:30:00 | 370.10 | 2024-03-22 10:40:00 | 368.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-03-27 09:50:00 | 364.55 | 2024-03-27 10:40:00 | 365.62 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-28 10:55:00 | 360.00 | 2024-03-28 13:15:00 | 360.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-04-02 09:35:00 | 378.30 | 2024-04-02 09:45:00 | 377.27 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-04-03 09:50:00 | 388.35 | 2024-04-03 09:55:00 | 386.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-04-04 09:50:00 | 383.95 | 2024-04-04 11:00:00 | 382.31 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-04-04 09:50:00 | 383.95 | 2024-04-04 15:20:00 | 383.60 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-04-05 10:25:00 | 386.30 | 2024-04-05 10:40:00 | 387.96 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-04-05 10:25:00 | 386.30 | 2024-04-05 11:25:00 | 388.20 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-04-16 11:05:00 | 369.60 | 2024-04-16 11:50:00 | 371.15 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-04-16 11:05:00 | 369.60 | 2024-04-16 15:20:00 | 373.45 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2024-04-22 09:35:00 | 376.25 | 2024-04-22 09:40:00 | 374.85 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-26 09:50:00 | 390.80 | 2024-04-26 09:55:00 | 389.58 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-03 09:30:00 | 401.55 | 2024-05-03 09:35:00 | 400.26 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-06 09:35:00 | 388.75 | 2024-05-06 09:40:00 | 386.93 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-06 09:35:00 | 388.75 | 2024-05-06 10:30:00 | 388.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 10:35:00 | 381.10 | 2024-05-07 11:00:00 | 379.24 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-07 10:35:00 | 381.10 | 2024-05-07 13:40:00 | 380.40 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2024-05-08 11:05:00 | 378.60 | 2024-05-08 11:15:00 | 379.95 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-05-08 11:05:00 | 378.60 | 2024-05-08 14:25:00 | 378.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 10:15:00 | 378.75 | 2024-05-09 10:40:00 | 377.41 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-09 10:15:00 | 378.75 | 2024-05-09 11:35:00 | 378.75 | STOP_HIT | 0.50 | 0.00% |
