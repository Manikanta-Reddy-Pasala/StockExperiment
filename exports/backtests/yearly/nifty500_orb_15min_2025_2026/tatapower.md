# Tata Power Co. Ltd. (TATAPOWER)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 435.50
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
| ENTRY1 | 97 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 14 |
| STOP_HIT | 83 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 83
- **Target hits / Stop hits / Partials:** 14 / 83 / 38
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 12.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 22 | 33.3% | 6 | 44 | 16 | 0.08% | 5.0% |
| BUY @ 2nd Alert (retest1) | 66 | 22 | 33.3% | 6 | 44 | 16 | 0.08% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 69 | 30 | 43.5% | 8 | 39 | 22 | 0.11% | 7.6% |
| SELL @ 2nd Alert (retest1) | 69 | 30 | 43.5% | 8 | 39 | 22 | 0.11% | 7.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 135 | 52 | 38.5% | 14 | 83 | 38 | 0.09% | 12.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 394.15 | 392.45 | 0.00 | ORB-long ORB[390.30,394.10] vol=1.9x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 10:40:00 | 395.92 | 394.01 | 0.00 | T1 1.5R @ 395.92 |
| Stop hit — per-position SL triggered | 2025-05-14 10:55:00 | 394.15 | 394.06 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 411.65 | 410.10 | 0.00 | ORB-long ORB[406.90,411.30] vol=1.6x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-05-19 09:35:00 | 410.42 | 410.05 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:45:00 | 399.45 | 401.32 | 0.00 | ORB-short ORB[400.95,403.25] vol=1.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-05-27 10:20:00 | 400.32 | 400.68 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:45:00 | 394.25 | 396.83 | 0.00 | ORB-short ORB[397.00,399.90] vol=1.5x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-05-30 09:50:00 | 395.30 | 396.64 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 395.70 | 396.34 | 0.00 | ORB-short ORB[396.10,399.15] vol=6.7x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:35:00 | 394.10 | 396.06 | 0.00 | T1 1.5R @ 394.10 |
| Target hit | 2025-06-03 15:20:00 | 391.45 | 393.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:35:00 | 390.90 | 391.63 | 0.00 | ORB-short ORB[391.00,393.80] vol=2.1x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-06-04 12:35:00 | 392.00 | 391.58 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:45:00 | 397.50 | 394.68 | 0.00 | ORB-long ORB[392.10,395.50] vol=3.4x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-06-06 12:20:00 | 396.31 | 395.80 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:45:00 | 409.70 | 410.85 | 0.00 | ORB-short ORB[410.10,412.25] vol=2.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-06-12 09:50:00 | 410.75 | 410.84 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 393.80 | 396.37 | 0.00 | ORB-short ORB[395.30,398.60] vol=1.5x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:45:00 | 392.29 | 395.05 | 0.00 | T1 1.5R @ 392.29 |
| Stop hit — per-position SL triggered | 2025-06-16 10:05:00 | 393.80 | 394.52 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 401.75 | 400.28 | 0.00 | ORB-long ORB[398.55,400.80] vol=1.6x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-06-17 09:45:00 | 400.86 | 400.61 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 396.90 | 395.87 | 0.00 | ORB-long ORB[393.80,396.65] vol=1.9x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-06-18 10:20:00 | 395.94 | 396.40 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:30:00 | 390.60 | 392.10 | 0.00 | ORB-short ORB[391.80,394.10] vol=1.6x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:05:00 | 389.20 | 391.51 | 0.00 | T1 1.5R @ 389.20 |
| Target hit | 2025-06-19 15:20:00 | 384.20 | 387.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:45:00 | 386.10 | 383.83 | 0.00 | ORB-long ORB[381.60,386.00] vol=1.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-06-20 09:55:00 | 384.89 | 384.05 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:50:00 | 398.85 | 396.21 | 0.00 | ORB-long ORB[394.85,396.75] vol=1.9x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-06-24 09:55:00 | 397.71 | 396.41 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:45:00 | 402.65 | 400.91 | 0.00 | ORB-long ORB[398.65,401.70] vol=4.0x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-07-04 11:20:00 | 401.84 | 401.19 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:30:00 | 405.50 | 404.35 | 0.00 | ORB-long ORB[400.80,405.40] vol=2.1x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-07-08 09:35:00 | 404.60 | 404.42 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:15:00 | 403.80 | 401.23 | 0.00 | ORB-long ORB[399.35,402.05] vol=1.8x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 11:25:00 | 404.84 | 401.65 | 0.00 | T1 1.5R @ 404.84 |
| Stop hit — per-position SL triggered | 2025-07-09 11:45:00 | 403.80 | 401.99 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:20:00 | 397.75 | 398.34 | 0.00 | ORB-short ORB[399.00,401.70] vol=9.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:30:00 | 396.50 | 398.27 | 0.00 | T1 1.5R @ 396.50 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 397.75 | 398.09 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:50:00 | 402.00 | 399.71 | 0.00 | ORB-long ORB[395.50,399.30] vol=1.7x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-07-14 11:20:00 | 400.97 | 399.97 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:40:00 | 400.55 | 403.25 | 0.00 | ORB-short ORB[402.10,405.30] vol=1.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:55:00 | 399.17 | 402.17 | 0.00 | T1 1.5R @ 399.17 |
| Stop hit — per-position SL triggered | 2025-07-15 10:05:00 | 400.55 | 401.98 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 405.00 | 404.05 | 0.00 | ORB-long ORB[402.50,404.90] vol=1.8x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 09:35:00 | 406.31 | 404.36 | 0.00 | T1 1.5R @ 406.31 |
| Stop hit — per-position SL triggered | 2025-07-16 09:40:00 | 405.00 | 404.39 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:05:00 | 409.55 | 412.18 | 0.00 | ORB-short ORB[411.00,413.80] vol=1.8x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 408.16 | 410.86 | 0.00 | T1 1.5R @ 408.16 |
| Stop hit — per-position SL triggered | 2025-07-18 11:25:00 | 409.55 | 409.77 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:55:00 | 399.25 | 401.50 | 0.00 | ORB-short ORB[401.50,403.90] vol=2.0x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-07-22 10:10:00 | 400.07 | 401.22 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 11:00:00 | 403.10 | 401.50 | 0.00 | ORB-long ORB[399.65,402.70] vol=2.0x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:15:00 | 404.26 | 401.92 | 0.00 | T1 1.5R @ 404.26 |
| Stop hit — per-position SL triggered | 2025-07-23 11:30:00 | 403.10 | 402.06 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:15:00 | 400.40 | 401.40 | 0.00 | ORB-short ORB[401.60,403.95] vol=1.5x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-07-24 10:50:00 | 401.09 | 401.07 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 398.60 | 399.76 | 0.00 | ORB-short ORB[399.00,401.25] vol=2.5x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-07-25 09:45:00 | 399.33 | 399.68 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-07-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:40:00 | 399.35 | 398.25 | 0.00 | ORB-long ORB[393.50,398.50] vol=3.1x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-07-28 10:45:00 | 398.38 | 398.29 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 11:05:00 | 396.65 | 399.08 | 0.00 | ORB-short ORB[397.85,401.30] vol=3.5x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 397.63 | 398.96 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 400.05 | 398.15 | 0.00 | ORB-long ORB[396.35,399.70] vol=1.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 399.18 | 398.33 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:30:00 | 394.05 | 395.96 | 0.00 | ORB-short ORB[395.10,398.15] vol=1.9x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 09:50:00 | 392.30 | 394.63 | 0.00 | T1 1.5R @ 392.30 |
| Target hit | 2025-08-01 10:30:00 | 393.80 | 393.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2025-08-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:35:00 | 382.10 | 383.85 | 0.00 | ORB-short ORB[384.25,387.00] vol=3.3x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-08-06 10:45:00 | 382.99 | 383.75 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 382.85 | 381.12 | 0.00 | ORB-long ORB[377.50,381.50] vol=1.9x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-08-11 11:40:00 | 381.97 | 381.25 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:15:00 | 388.45 | 386.66 | 0.00 | ORB-long ORB[383.50,386.65] vol=1.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-08-12 10:25:00 | 387.56 | 386.73 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-08-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 11:05:00 | 387.05 | 385.79 | 0.00 | ORB-long ORB[384.70,386.75] vol=2.4x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-08-13 11:40:00 | 386.37 | 386.22 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 384.45 | 386.17 | 0.00 | ORB-short ORB[385.60,388.60] vol=2.1x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 09:35:00 | 383.39 | 385.60 | 0.00 | T1 1.5R @ 383.39 |
| Stop hit — per-position SL triggered | 2025-08-14 09:40:00 | 384.45 | 385.51 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:40:00 | 391.50 | 390.37 | 0.00 | ORB-long ORB[388.05,391.40] vol=2.2x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-08-20 09:55:00 | 390.75 | 390.72 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-08-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:10:00 | 385.40 | 386.40 | 0.00 | ORB-short ORB[386.55,388.70] vol=1.7x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-08-22 11:55:00 | 385.97 | 386.08 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 379.95 | 380.85 | 0.00 | ORB-short ORB[380.00,383.60] vol=1.9x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:35:00 | 378.89 | 380.56 | 0.00 | T1 1.5R @ 378.89 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 379.95 | 380.11 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:00:00 | 377.75 | 377.22 | 0.00 | ORB-long ORB[374.15,377.60] vol=1.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-09-01 11:05:00 | 377.13 | 377.22 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:45:00 | 384.00 | 382.08 | 0.00 | ORB-long ORB[380.00,383.55] vol=1.8x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:10:00 | 385.27 | 382.79 | 0.00 | T1 1.5R @ 385.27 |
| Target hit | 2025-09-02 14:00:00 | 385.50 | 385.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-09-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 11:00:00 | 385.75 | 386.88 | 0.00 | ORB-short ORB[385.80,388.05] vol=2.0x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-09-11 11:10:00 | 386.39 | 386.82 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:55:00 | 387.00 | 388.37 | 0.00 | ORB-short ORB[388.20,389.80] vol=1.6x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:30:00 | 386.22 | 387.78 | 0.00 | T1 1.5R @ 386.22 |
| Target hit | 2025-09-12 14:35:00 | 386.85 | 386.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 396.65 | 393.85 | 0.00 | ORB-long ORB[388.00,393.50] vol=6.1x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:35:00 | 398.18 | 395.29 | 0.00 | T1 1.5R @ 398.18 |
| Stop hit — per-position SL triggered | 2025-09-16 10:35:00 | 396.65 | 397.04 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:55:00 | 393.30 | 394.09 | 0.00 | ORB-short ORB[393.50,396.40] vol=3.0x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:10:00 | 392.15 | 393.90 | 0.00 | T1 1.5R @ 392.15 |
| Stop hit — per-position SL triggered | 2025-09-24 12:30:00 | 393.30 | 393.48 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-09-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:00:00 | 387.80 | 386.74 | 0.00 | ORB-long ORB[384.25,386.60] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-09-29 10:10:00 | 386.94 | 386.76 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-09-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 11:10:00 | 391.25 | 389.76 | 0.00 | ORB-long ORB[387.10,391.00] vol=3.9x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-09-30 11:55:00 | 390.29 | 389.87 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 390.75 | 391.69 | 0.00 | ORB-short ORB[392.80,395.95] vol=4.4x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 391.53 | 391.65 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:05:00 | 393.20 | 394.46 | 0.00 | ORB-short ORB[394.10,396.85] vol=1.8x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-10-07 11:10:00 | 394.03 | 394.38 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:05:00 | 390.35 | 391.14 | 0.00 | ORB-short ORB[391.25,393.40] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:00:00 | 389.14 | 390.57 | 0.00 | T1 1.5R @ 389.14 |
| Target hit | 2025-10-08 12:45:00 | 389.80 | 389.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — SELL (started 2025-10-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:35:00 | 386.85 | 387.95 | 0.00 | ORB-short ORB[387.25,392.05] vol=2.5x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:00:00 | 385.48 | 387.32 | 0.00 | T1 1.5R @ 385.48 |
| Stop hit — per-position SL triggered | 2025-10-13 10:30:00 | 386.85 | 387.02 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:10:00 | 389.25 | 390.13 | 0.00 | ORB-short ORB[391.05,393.10] vol=5.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-10-14 11:40:00 | 390.09 | 389.88 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:00:00 | 393.90 | 392.84 | 0.00 | ORB-long ORB[391.30,393.30] vol=1.8x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:50:00 | 395.34 | 393.75 | 0.00 | T1 1.5R @ 395.34 |
| Target hit | 2025-10-15 15:20:00 | 397.10 | 394.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-10-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:50:00 | 399.45 | 398.14 | 0.00 | ORB-long ORB[396.65,399.30] vol=2.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 14:45:00 | 400.56 | 399.35 | 0.00 | T1 1.5R @ 400.56 |
| Stop hit — per-position SL triggered | 2025-10-20 15:00:00 | 399.45 | 399.38 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-10-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:35:00 | 402.30 | 400.61 | 0.00 | ORB-long ORB[398.15,401.25] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 11:00:00 | 403.51 | 401.39 | 0.00 | T1 1.5R @ 403.51 |
| Stop hit — per-position SL triggered | 2025-10-23 11:05:00 | 402.30 | 401.42 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:10:00 | 400.10 | 399.16 | 0.00 | ORB-long ORB[395.65,399.15] vol=1.9x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-10-27 10:20:00 | 399.19 | 399.19 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 405.35 | 404.17 | 0.00 | ORB-long ORB[398.70,404.60] vol=4.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-10-29 09:40:00 | 404.25 | 404.33 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:05:00 | 396.35 | 398.81 | 0.00 | ORB-short ORB[398.40,402.30] vol=1.8x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:20:00 | 395.10 | 397.84 | 0.00 | T1 1.5R @ 395.10 |
| Stop hit — per-position SL triggered | 2025-11-06 11:45:00 | 396.35 | 396.54 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:10:00 | 391.80 | 393.33 | 0.00 | ORB-short ORB[392.65,395.80] vol=1.9x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:35:00 | 390.53 | 392.99 | 0.00 | T1 1.5R @ 390.53 |
| Stop hit — per-position SL triggered | 2025-11-11 10:50:00 | 391.80 | 392.76 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:50:00 | 392.20 | 388.20 | 0.00 | ORB-long ORB[385.00,390.00] vol=2.1x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 391.24 | 388.94 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:45:00 | 389.20 | 387.82 | 0.00 | ORB-long ORB[385.75,388.95] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-11-14 09:50:00 | 388.20 | 388.12 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:15:00 | 385.40 | 387.98 | 0.00 | ORB-short ORB[386.65,391.50] vol=2.3x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:50:00 | 384.32 | 387.39 | 0.00 | T1 1.5R @ 384.32 |
| Target hit | 2025-11-24 15:20:00 | 382.20 | 384.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-11-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:45:00 | 386.20 | 383.82 | 0.00 | ORB-long ORB[380.30,384.85] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-11-26 09:55:00 | 385.23 | 384.02 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:20:00 | 384.85 | 385.60 | 0.00 | ORB-short ORB[387.00,389.10] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 385.47 | 385.01 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:20:00 | 385.65 | 383.63 | 0.00 | ORB-long ORB[382.00,384.65] vol=2.2x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-12-04 10:45:00 | 384.87 | 384.19 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:25:00 | 379.95 | 381.82 | 0.00 | ORB-short ORB[382.10,385.25] vol=1.8x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:55:00 | 378.65 | 381.24 | 0.00 | T1 1.5R @ 378.65 |
| Target hit | 2025-12-08 15:20:00 | 373.45 | 376.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 380.95 | 378.93 | 0.00 | ORB-long ORB[376.50,379.85] vol=2.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-12-10 11:00:00 | 379.89 | 380.89 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:55:00 | 381.90 | 379.02 | 0.00 | ORB-long ORB[376.20,379.85] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-12-11 11:05:00 | 381.16 | 379.87 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:35:00 | 374.50 | 375.90 | 0.00 | ORB-short ORB[375.15,377.65] vol=1.8x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-12-18 10:10:00 | 375.30 | 375.41 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:05:00 | 374.75 | 375.70 | 0.00 | ORB-short ORB[375.00,377.00] vol=2.2x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:30:00 | 373.82 | 375.58 | 0.00 | T1 1.5R @ 373.82 |
| Stop hit — per-position SL triggered | 2025-12-19 12:00:00 | 374.75 | 375.46 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 383.50 | 382.59 | 0.00 | ORB-long ORB[381.10,383.15] vol=2.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-12-24 09:45:00 | 382.84 | 382.70 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:55:00 | 384.55 | 383.76 | 0.00 | ORB-long ORB[381.25,384.50] vol=3.1x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:25:00 | 385.81 | 384.31 | 0.00 | T1 1.5R @ 385.81 |
| Target hit | 2026-01-02 15:20:00 | 393.30 | 389.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 11:15:00 | 383.85 | 384.90 | 0.00 | ORB-short ORB[384.40,387.90] vol=1.6x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 11:35:00 | 382.84 | 384.58 | 0.00 | T1 1.5R @ 382.84 |
| Target hit | 2026-01-07 15:20:00 | 380.65 | 381.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2026-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:30:00 | 370.00 | 371.64 | 0.00 | ORB-short ORB[370.60,374.00] vol=2.1x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:35:00 | 368.32 | 371.22 | 0.00 | T1 1.5R @ 368.32 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 370.00 | 370.69 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 10:25:00 | 366.15 | 363.32 | 0.00 | ORB-long ORB[360.25,364.80] vol=2.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-01-12 10:45:00 | 364.70 | 363.57 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:45:00 | 368.50 | 369.10 | 0.00 | ORB-short ORB[369.05,372.20] vol=2.1x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-01-13 10:05:00 | 369.57 | 369.02 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-01-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:45:00 | 369.00 | 367.88 | 0.00 | ORB-long ORB[366.30,368.65] vol=4.0x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 368.14 | 368.25 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 359.20 | 361.41 | 0.00 | ORB-short ORB[360.50,364.15] vol=1.8x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-01-20 09:40:00 | 360.24 | 361.21 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:40:00 | 369.65 | 367.44 | 0.00 | ORB-long ORB[363.60,368.00] vol=1.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-04 09:50:00 | 368.59 | 367.69 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:00:00 | 373.55 | 375.31 | 0.00 | ORB-short ORB[374.45,377.40] vol=1.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 372.29 | 375.09 | 0.00 | T1 1.5R @ 372.29 |
| Stop hit — per-position SL triggered | 2026-02-12 12:45:00 | 373.55 | 373.92 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-02-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:20:00 | 372.35 | 374.12 | 0.00 | ORB-short ORB[374.70,377.80] vol=3.7x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-02-13 10:35:00 | 373.43 | 373.88 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 377.70 | 376.07 | 0.00 | ORB-long ORB[370.90,376.10] vol=1.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:35:00 | 378.93 | 376.49 | 0.00 | T1 1.5R @ 378.93 |
| Stop hit — per-position SL triggered | 2026-02-16 11:50:00 | 377.70 | 376.58 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 373.85 | 371.70 | 0.00 | ORB-long ORB[369.25,372.20] vol=1.7x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:40:00 | 375.27 | 372.93 | 0.00 | T1 1.5R @ 375.27 |
| Target hit | 2026-02-20 15:20:00 | 377.40 | 375.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 378.75 | 380.73 | 0.00 | ORB-short ORB[378.90,383.65] vol=1.5x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:30:00 | 377.51 | 380.27 | 0.00 | T1 1.5R @ 377.51 |
| Stop hit — per-position SL triggered | 2026-02-25 13:30:00 | 378.75 | 379.67 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:35:00 | 383.40 | 382.16 | 0.00 | ORB-long ORB[380.05,382.45] vol=5.3x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 382.49 | 382.53 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-03-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:05:00 | 380.25 | 378.03 | 0.00 | ORB-long ORB[375.90,379.50] vol=1.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:25:00 | 382.09 | 379.31 | 0.00 | T1 1.5R @ 382.09 |
| Target hit | 2026-03-10 11:45:00 | 380.45 | 380.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 86 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 397.05 | 393.50 | 0.00 | ORB-long ORB[391.00,394.00] vol=2.0x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:40:00 | 399.77 | 395.17 | 0.00 | T1 1.5R @ 399.77 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 397.05 | 395.98 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 11:05:00 | 385.50 | 382.55 | 0.00 | ORB-long ORB[378.45,383.00] vol=2.8x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-04-07 11:10:00 | 384.36 | 382.67 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 419.80 | 414.54 | 0.00 | ORB-long ORB[410.30,416.30] vol=3.2x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-04-15 09:55:00 | 418.14 | 415.22 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 419.40 | 422.71 | 0.00 | ORB-short ORB[421.75,427.40] vol=1.7x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 420.82 | 422.15 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 426.90 | 427.58 | 0.00 | ORB-short ORB[427.15,432.00] vol=1.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-04-17 13:35:00 | 428.05 | 427.33 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 431.50 | 434.23 | 0.00 | ORB-short ORB[432.40,437.80] vol=3.9x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-04-23 11:25:00 | 432.45 | 434.10 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2026-04-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:05:00 | 427.50 | 429.10 | 0.00 | ORB-short ORB[428.70,433.35] vol=1.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 428.74 | 429.06 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 440.80 | 438.06 | 0.00 | ORB-long ORB[435.20,439.20] vol=2.0x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 443.20 | 439.50 | 0.00 | T1 1.5R @ 443.20 |
| Target hit | 2026-04-27 15:20:00 | 453.90 | 449.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 94 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 459.40 | 457.17 | 0.00 | ORB-long ORB[454.00,458.40] vol=2.5x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-04-28 10:05:00 | 457.83 | 458.47 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2026-05-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:35:00 | 443.00 | 444.92 | 0.00 | ORB-short ORB[444.00,447.80] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-05-04 10:45:00 | 444.51 | 444.70 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 441.80 | 440.35 | 0.00 | ORB-long ORB[437.25,441.40] vol=2.3x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:50:00 | 443.59 | 441.45 | 0.00 | T1 1.5R @ 443.59 |
| Stop hit — per-position SL triggered | 2026-05-05 10:35:00 | 441.80 | 442.61 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 440.15 | 441.61 | 0.00 | ORB-short ORB[442.20,446.00] vol=2.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-05-07 10:25:00 | 441.31 | 441.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:30:00 | 394.15 | 2025-05-14 10:40:00 | 395.92 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-14 09:30:00 | 394.15 | 2025-05-14 10:55:00 | 394.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-19 09:30:00 | 411.65 | 2025-05-19 09:35:00 | 410.42 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-05-27 09:45:00 | 399.45 | 2025-05-27 10:20:00 | 400.32 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-30 09:45:00 | 394.25 | 2025-05-30 09:50:00 | 395.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-03 09:35:00 | 395.70 | 2025-06-03 10:35:00 | 394.10 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-06-03 09:35:00 | 395.70 | 2025-06-03 15:20:00 | 391.45 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2025-06-04 10:35:00 | 390.90 | 2025-06-04 12:35:00 | 392.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-06 10:45:00 | 397.50 | 2025-06-06 12:20:00 | 396.31 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-12 09:45:00 | 409.70 | 2025-06-12 09:50:00 | 410.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-16 09:30:00 | 393.80 | 2025-06-16 09:45:00 | 392.29 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-16 09:30:00 | 393.80 | 2025-06-16 10:05:00 | 393.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-17 09:30:00 | 401.75 | 2025-06-17 09:45:00 | 400.86 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-18 09:30:00 | 396.90 | 2025-06-18 10:20:00 | 395.94 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-19 10:30:00 | 390.60 | 2025-06-19 11:05:00 | 389.20 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-19 10:30:00 | 390.60 | 2025-06-19 15:20:00 | 384.20 | TARGET_HIT | 0.50 | 1.64% |
| BUY | retest1 | 2025-06-20 09:45:00 | 386.10 | 2025-06-20 09:55:00 | 384.89 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-24 09:50:00 | 398.85 | 2025-06-24 09:55:00 | 397.71 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-04 10:45:00 | 402.65 | 2025-07-04 11:20:00 | 401.84 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-08 09:30:00 | 405.50 | 2025-07-08 09:35:00 | 404.60 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-09 11:15:00 | 403.80 | 2025-07-09 11:25:00 | 404.84 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-07-09 11:15:00 | 403.80 | 2025-07-09 11:45:00 | 403.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:20:00 | 397.75 | 2025-07-11 10:30:00 | 396.50 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-11 10:20:00 | 397.75 | 2025-07-11 11:10:00 | 397.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 10:50:00 | 402.00 | 2025-07-14 11:20:00 | 400.97 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-15 09:40:00 | 400.55 | 2025-07-15 09:55:00 | 399.17 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-15 09:40:00 | 400.55 | 2025-07-15 10:05:00 | 400.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 09:30:00 | 405.00 | 2025-07-16 09:35:00 | 406.31 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-07-16 09:30:00 | 405.00 | 2025-07-16 09:40:00 | 405.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:05:00 | 409.55 | 2025-07-18 10:15:00 | 408.16 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-18 10:05:00 | 409.55 | 2025-07-18 11:25:00 | 409.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 09:55:00 | 399.25 | 2025-07-22 10:10:00 | 400.07 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-23 11:00:00 | 403.10 | 2025-07-23 11:15:00 | 404.26 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-07-23 11:00:00 | 403.10 | 2025-07-23 11:30:00 | 403.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 10:15:00 | 400.40 | 2025-07-24 10:50:00 | 401.09 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-25 09:40:00 | 398.60 | 2025-07-25 09:45:00 | 399.33 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-28 10:40:00 | 399.35 | 2025-07-28 10:45:00 | 398.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-29 11:05:00 | 396.65 | 2025-07-29 11:15:00 | 397.63 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-31 11:00:00 | 400.05 | 2025-07-31 11:15:00 | 399.18 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-01 09:30:00 | 394.05 | 2025-08-01 09:50:00 | 392.30 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-01 09:30:00 | 394.05 | 2025-08-01 10:30:00 | 393.80 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-08-06 10:35:00 | 382.10 | 2025-08-06 10:45:00 | 382.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-11 11:10:00 | 382.85 | 2025-08-11 11:40:00 | 381.97 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-12 10:15:00 | 388.45 | 2025-08-12 10:25:00 | 387.56 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-13 11:05:00 | 387.05 | 2025-08-13 11:40:00 | 386.37 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-14 09:30:00 | 384.45 | 2025-08-14 09:35:00 | 383.39 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-14 09:30:00 | 384.45 | 2025-08-14 09:40:00 | 384.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 09:40:00 | 391.50 | 2025-08-20 09:55:00 | 390.75 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-22 11:10:00 | 385.40 | 2025-08-22 11:55:00 | 385.97 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-08-26 09:30:00 | 379.95 | 2025-08-26 09:35:00 | 378.89 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-26 09:30:00 | 379.95 | 2025-08-26 09:55:00 | 379.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 11:00:00 | 377.75 | 2025-09-01 11:05:00 | 377.13 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-02 09:45:00 | 384.00 | 2025-09-02 10:10:00 | 385.27 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-02 09:45:00 | 384.00 | 2025-09-02 14:00:00 | 385.50 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-09-11 11:00:00 | 385.75 | 2025-09-11 11:10:00 | 386.39 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-12 10:55:00 | 387.00 | 2025-09-12 11:30:00 | 386.22 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-09-12 10:55:00 | 387.00 | 2025-09-12 14:35:00 | 386.85 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2025-09-16 09:30:00 | 396.65 | 2025-09-16 09:35:00 | 398.18 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-16 09:30:00 | 396.65 | 2025-09-16 10:35:00 | 396.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 10:55:00 | 393.30 | 2025-09-24 11:10:00 | 392.15 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-24 10:55:00 | 393.30 | 2025-09-24 12:30:00 | 393.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-29 10:00:00 | 387.80 | 2025-09-29 10:10:00 | 386.94 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-30 11:10:00 | 391.25 | 2025-09-30 11:55:00 | 390.29 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-06 11:10:00 | 390.75 | 2025-10-06 11:15:00 | 391.53 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-07 11:05:00 | 393.20 | 2025-10-07 11:10:00 | 394.03 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-08 10:05:00 | 390.35 | 2025-10-08 11:00:00 | 389.14 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-08 10:05:00 | 390.35 | 2025-10-08 12:45:00 | 389.80 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-10-13 09:35:00 | 386.85 | 2025-10-13 10:00:00 | 385.48 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-13 09:35:00 | 386.85 | 2025-10-13 10:30:00 | 386.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 11:10:00 | 389.25 | 2025-10-14 11:40:00 | 390.09 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-15 10:00:00 | 393.90 | 2025-10-15 11:50:00 | 395.34 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-15 10:00:00 | 393.90 | 2025-10-15 15:20:00 | 397.10 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2025-10-20 10:50:00 | 399.45 | 2025-10-20 14:45:00 | 400.56 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-20 10:50:00 | 399.45 | 2025-10-20 15:00:00 | 399.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-23 10:35:00 | 402.30 | 2025-10-23 11:00:00 | 403.51 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-23 10:35:00 | 402.30 | 2025-10-23 11:05:00 | 402.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 10:10:00 | 400.10 | 2025-10-27 10:20:00 | 399.19 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-29 09:30:00 | 405.35 | 2025-10-29 09:40:00 | 404.25 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-06 10:05:00 | 396.35 | 2025-11-06 10:20:00 | 395.10 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-11-06 10:05:00 | 396.35 | 2025-11-06 11:45:00 | 396.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:10:00 | 391.80 | 2025-11-11 10:35:00 | 390.53 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-11-11 10:10:00 | 391.80 | 2025-11-11 10:50:00 | 391.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 10:50:00 | 392.20 | 2025-11-13 11:15:00 | 391.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-14 09:45:00 | 389.20 | 2025-11-14 09:50:00 | 388.20 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-24 11:15:00 | 385.40 | 2025-11-24 11:50:00 | 384.32 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-24 11:15:00 | 385.40 | 2025-11-24 15:20:00 | 382.20 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2025-11-26 09:45:00 | 386.20 | 2025-11-26 09:55:00 | 385.23 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-03 10:20:00 | 384.85 | 2025-12-03 12:15:00 | 385.47 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-04 10:20:00 | 385.65 | 2025-12-04 10:45:00 | 384.87 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-08 10:25:00 | 379.95 | 2025-12-08 10:55:00 | 378.65 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-08 10:25:00 | 379.95 | 2025-12-08 15:20:00 | 373.45 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2025-12-10 09:30:00 | 380.95 | 2025-12-10 11:00:00 | 379.89 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-11 10:55:00 | 381.90 | 2025-12-11 11:05:00 | 381.16 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-18 09:35:00 | 374.50 | 2025-12-18 10:10:00 | 375.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-19 11:05:00 | 374.75 | 2025-12-19 11:30:00 | 373.82 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-19 11:05:00 | 374.75 | 2025-12-19 12:00:00 | 374.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-24 09:30:00 | 383.50 | 2025-12-24 09:45:00 | 382.84 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-02 09:55:00 | 384.55 | 2026-01-02 10:25:00 | 385.81 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-01-02 09:55:00 | 384.55 | 2026-01-02 15:20:00 | 393.30 | TARGET_HIT | 0.50 | 2.28% |
| SELL | retest1 | 2026-01-07 11:15:00 | 383.85 | 2026-01-07 11:35:00 | 382.84 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-01-07 11:15:00 | 383.85 | 2026-01-07 15:20:00 | 380.65 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2026-01-09 09:30:00 | 370.00 | 2026-01-09 09:35:00 | 368.32 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-09 09:30:00 | 370.00 | 2026-01-09 09:45:00 | 370.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-12 10:25:00 | 366.15 | 2026-01-12 10:45:00 | 364.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-13 09:45:00 | 368.50 | 2026-01-13 10:05:00 | 369.57 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-14 10:45:00 | 369.00 | 2026-01-14 12:15:00 | 368.14 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-20 09:35:00 | 359.20 | 2026-01-20 09:40:00 | 360.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-04 09:40:00 | 369.65 | 2026-02-04 09:50:00 | 368.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-12 11:00:00 | 373.55 | 2026-02-12 11:15:00 | 372.29 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-12 11:00:00 | 373.55 | 2026-02-12 12:45:00 | 373.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:20:00 | 372.35 | 2026-02-13 10:35:00 | 373.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-16 11:15:00 | 377.70 | 2026-02-16 11:35:00 | 378.93 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-16 11:15:00 | 377.70 | 2026-02-16 11:50:00 | 377.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:45:00 | 373.85 | 2026-02-20 12:40:00 | 375.27 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-20 10:45:00 | 373.85 | 2026-02-20 15:20:00 | 377.40 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2026-02-25 11:05:00 | 378.75 | 2026-02-25 12:30:00 | 377.51 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-25 11:05:00 | 378.75 | 2026-02-25 13:30:00 | 378.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:35:00 | 383.40 | 2026-02-26 11:35:00 | 382.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-10 10:05:00 | 380.25 | 2026-03-10 10:25:00 | 382.09 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-10 10:05:00 | 380.25 | 2026-03-10 11:45:00 | 380.45 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2026-03-17 09:35:00 | 397.05 | 2026-03-17 09:40:00 | 399.77 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-17 09:35:00 | 397.05 | 2026-03-17 09:55:00 | 397.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 11:05:00 | 385.50 | 2026-04-07 11:10:00 | 384.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-15 09:50:00 | 419.80 | 2026-04-15 09:55:00 | 418.14 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-16 11:00:00 | 419.40 | 2026-04-16 11:40:00 | 420.82 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-17 11:00:00 | 426.90 | 2026-04-17 13:35:00 | 428.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-23 11:10:00 | 431.50 | 2026-04-23 11:25:00 | 432.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-24 11:05:00 | 427.50 | 2026-04-24 11:20:00 | 428.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-27 09:45:00 | 440.80 | 2026-04-27 09:50:00 | 443.20 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-27 09:45:00 | 440.80 | 2026-04-27 15:20:00 | 453.90 | TARGET_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2026-04-28 09:30:00 | 459.40 | 2026-04-28 10:05:00 | 457.83 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-04 10:35:00 | 443.00 | 2026-05-04 10:45:00 | 444.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-05 09:35:00 | 441.80 | 2026-05-05 09:50:00 | 443.59 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-05 09:35:00 | 441.80 | 2026-05-05 10:35:00 | 441.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 10:15:00 | 440.15 | 2026-05-07 10:25:00 | 441.31 | STOP_HIT | 1.00 | -0.26% |
