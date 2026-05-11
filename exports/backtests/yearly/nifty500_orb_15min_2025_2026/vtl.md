# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 583.10
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
| ENTRY1 | 60 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 14 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 82 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 46
- **Target hits / Stop hits / Partials:** 14 / 46 / 22
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 15.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 12 | 34.3% | 4 | 23 | 8 | 0.06% | 2.0% |
| BUY @ 2nd Alert (retest1) | 35 | 12 | 34.3% | 4 | 23 | 8 | 0.06% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 24 | 51.1% | 10 | 23 | 14 | 0.29% | 13.4% |
| SELL @ 2nd Alert (retest1) | 47 | 24 | 51.1% | 10 | 23 | 14 | 0.29% | 13.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 82 | 36 | 43.9% | 14 | 46 | 22 | 0.19% | 15.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:35:00 | 483.25 | 481.17 | 0.00 | ORB-long ORB[477.25,483.00] vol=2.8x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-05-14 09:45:00 | 481.36 | 481.28 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:55:00 | 483.25 | 480.24 | 0.00 | ORB-long ORB[477.70,481.15] vol=1.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-05-15 10:15:00 | 481.31 | 481.42 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:05:00 | 494.85 | 492.00 | 0.00 | ORB-long ORB[487.00,493.20] vol=1.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-05-28 10:25:00 | 493.24 | 492.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:25:00 | 498.50 | 500.13 | 0.00 | ORB-short ORB[499.45,504.80] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-05-29 10:50:00 | 499.95 | 499.93 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:30:00 | 509.70 | 505.45 | 0.00 | ORB-long ORB[501.00,507.00] vol=5.8x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-05-30 09:35:00 | 507.74 | 506.19 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 472.70 | 477.03 | 0.00 | ORB-short ORB[476.90,483.15] vol=4.4x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-06-16 10:30:00 | 474.61 | 474.66 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 478.30 | 476.29 | 0.00 | ORB-long ORB[472.70,477.70] vol=2.8x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 10:00:00 | 480.57 | 477.00 | 0.00 | T1 1.5R @ 480.57 |
| Target hit | 2025-06-17 12:00:00 | 480.00 | 480.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2025-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:35:00 | 463.15 | 466.46 | 0.00 | ORB-short ORB[464.60,468.50] vol=1.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:20:00 | 461.30 | 465.23 | 0.00 | T1 1.5R @ 461.30 |
| Target hit | 2025-06-19 15:20:00 | 455.10 | 459.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-07-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:00:00 | 502.65 | 498.32 | 0.00 | ORB-long ORB[497.20,501.85] vol=2.3x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 11:15:00 | 505.41 | 500.71 | 0.00 | T1 1.5R @ 505.41 |
| Stop hit — per-position SL triggered | 2025-07-03 12:55:00 | 502.65 | 502.11 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:55:00 | 503.20 | 498.88 | 0.00 | ORB-long ORB[495.05,500.95] vol=2.0x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-07-04 11:05:00 | 501.43 | 498.94 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:35:00 | 509.30 | 506.22 | 0.00 | ORB-long ORB[502.60,507.30] vol=1.7x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-07-11 10:40:00 | 507.48 | 507.47 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:00:00 | 500.25 | 502.79 | 0.00 | ORB-short ORB[501.15,505.60] vol=1.7x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-07-16 11:05:00 | 501.47 | 502.67 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:50:00 | 504.15 | 501.57 | 0.00 | ORB-long ORB[500.00,503.90] vol=2.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 502.96 | 501.73 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:25:00 | 497.25 | 502.14 | 0.00 | ORB-short ORB[500.60,505.40] vol=1.5x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-07-18 10:40:00 | 498.70 | 501.83 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:45:00 | 406.70 | 404.74 | 0.00 | ORB-long ORB[396.55,400.00] vol=17.1x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:50:00 | 409.08 | 404.93 | 0.00 | T1 1.5R @ 409.08 |
| Stop hit — per-position SL triggered | 2025-08-18 10:55:00 | 406.70 | 404.98 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 11:10:00 | 425.00 | 425.90 | 0.00 | ORB-short ORB[425.05,431.10] vol=2.1x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:50:00 | 423.46 | 425.76 | 0.00 | T1 1.5R @ 423.46 |
| Target hit | 2025-08-21 15:20:00 | 420.60 | 423.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 417.55 | 420.02 | 0.00 | ORB-short ORB[418.80,422.10] vol=2.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-08-22 10:10:00 | 419.18 | 419.15 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:15:00 | 405.40 | 407.76 | 0.00 | ORB-short ORB[407.10,411.85] vol=1.9x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 11:05:00 | 402.94 | 406.63 | 0.00 | T1 1.5R @ 402.94 |
| Target hit | 2025-08-26 15:20:00 | 398.95 | 402.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:15:00 | 421.80 | 425.81 | 0.00 | ORB-short ORB[423.55,429.80] vol=4.0x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-09-22 11:45:00 | 423.03 | 425.25 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 418.30 | 420.16 | 0.00 | ORB-short ORB[418.45,423.95] vol=1.7x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-09-23 09:35:00 | 419.87 | 420.12 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 415.35 | 417.83 | 0.00 | ORB-short ORB[416.00,419.60] vol=3.1x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-09-24 11:40:00 | 416.29 | 417.38 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 09:40:00 | 414.30 | 412.53 | 0.00 | ORB-long ORB[408.85,412.25] vol=2.4x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 412.28 | 414.02 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:05:00 | 409.80 | 410.97 | 0.00 | ORB-short ORB[410.05,413.85] vol=3.4x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:25:00 | 407.90 | 410.41 | 0.00 | T1 1.5R @ 407.90 |
| Target hit | 2025-10-03 15:20:00 | 408.00 | 408.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2025-10-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:55:00 | 405.50 | 407.95 | 0.00 | ORB-short ORB[407.00,411.20] vol=3.2x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 11:05:00 | 404.05 | 407.35 | 0.00 | T1 1.5R @ 404.05 |
| Stop hit — per-position SL triggered | 2025-10-06 12:35:00 | 405.50 | 406.39 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:05:00 | 404.85 | 402.79 | 0.00 | ORB-long ORB[398.00,403.90] vol=1.7x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-10-10 10:25:00 | 403.35 | 402.85 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-10-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:50:00 | 396.30 | 398.48 | 0.00 | ORB-short ORB[397.30,400.60] vol=1.9x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 394.65 | 396.71 | 0.00 | T1 1.5R @ 394.65 |
| Target hit | 2025-10-14 14:40:00 | 392.25 | 392.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:15:00 | 397.95 | 395.88 | 0.00 | ORB-long ORB[393.20,397.60] vol=2.4x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:30:00 | 399.54 | 396.28 | 0.00 | T1 1.5R @ 399.54 |
| Stop hit — per-position SL triggered | 2025-10-15 11:45:00 | 397.95 | 397.21 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:35:00 | 409.00 | 406.46 | 0.00 | ORB-long ORB[401.20,405.60] vol=2.3x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-10-16 09:45:00 | 407.55 | 406.79 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:10:00 | 405.90 | 407.25 | 0.00 | ORB-short ORB[407.30,412.00] vol=3.5x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-10-17 12:05:00 | 406.64 | 407.04 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:00:00 | 401.95 | 403.46 | 0.00 | ORB-short ORB[402.15,406.75] vol=1.8x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:45:00 | 400.42 | 402.88 | 0.00 | T1 1.5R @ 400.42 |
| Stop hit — per-position SL triggered | 2025-10-20 10:55:00 | 401.95 | 402.43 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:15:00 | 438.60 | 440.87 | 0.00 | ORB-short ORB[440.35,444.15] vol=4.5x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:55:00 | 436.86 | 440.26 | 0.00 | T1 1.5R @ 436.86 |
| Target hit | 2025-10-28 15:20:00 | 433.05 | 437.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 435.90 | 432.29 | 0.00 | ORB-long ORB[431.25,435.55] vol=2.2x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:30:00 | 438.34 | 438.16 | 0.00 | T1 1.5R @ 438.34 |
| Target hit | 2025-10-29 14:40:00 | 441.50 | 441.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-10-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:05:00 | 434.05 | 436.78 | 0.00 | ORB-short ORB[436.00,438.80] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-10-31 10:10:00 | 435.13 | 436.62 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-11-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:25:00 | 444.50 | 441.88 | 0.00 | ORB-long ORB[438.20,441.65] vol=7.4x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-11-03 10:30:00 | 443.15 | 442.18 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 433.75 | 435.21 | 0.00 | ORB-short ORB[434.25,440.00] vol=2.0x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:10:00 | 431.77 | 433.72 | 0.00 | T1 1.5R @ 431.77 |
| Target hit | 2025-11-06 15:00:00 | 431.25 | 430.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — BUY (started 2025-11-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 11:00:00 | 428.45 | 425.82 | 0.00 | ORB-long ORB[424.05,428.05] vol=2.5x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:30:00 | 429.92 | 426.10 | 0.00 | T1 1.5R @ 429.92 |
| Target hit | 2025-11-07 15:20:00 | 439.50 | 432.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-11-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:10:00 | 438.85 | 440.38 | 0.00 | ORB-short ORB[439.00,442.75] vol=3.3x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-11-11 12:10:00 | 440.27 | 440.27 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:30:00 | 455.50 | 451.37 | 0.00 | ORB-long ORB[447.00,451.90] vol=5.7x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-11-13 10:35:00 | 453.81 | 451.73 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 450.90 | 453.39 | 0.00 | ORB-short ORB[452.15,457.20] vol=1.8x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-11-17 10:40:00 | 452.44 | 452.24 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:00:00 | 459.85 | 457.12 | 0.00 | ORB-long ORB[454.40,458.60] vol=2.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:05:00 | 461.90 | 462.87 | 0.00 | T1 1.5R @ 461.90 |
| Target hit | 2025-11-19 10:45:00 | 463.00 | 464.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-11-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:05:00 | 459.80 | 462.83 | 0.00 | ORB-short ORB[460.55,464.00] vol=3.5x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:10:00 | 457.85 | 461.24 | 0.00 | T1 1.5R @ 457.85 |
| Target hit | 2025-11-21 15:20:00 | 451.65 | 459.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-11-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:10:00 | 443.35 | 446.61 | 0.00 | ORB-short ORB[448.60,454.20] vol=8.9x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:10:00 | 441.10 | 445.68 | 0.00 | T1 1.5R @ 441.10 |
| Target hit | 2025-11-24 15:20:00 | 434.80 | 440.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 427.25 | 428.96 | 0.00 | ORB-short ORB[427.65,430.60] vol=2.7x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-11-28 10:20:00 | 428.63 | 428.27 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:40:00 | 432.60 | 429.76 | 0.00 | ORB-long ORB[428.15,431.05] vol=1.8x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-12-02 11:20:00 | 431.55 | 430.39 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:25:00 | 426.85 | 428.70 | 0.00 | ORB-short ORB[427.45,431.95] vol=5.1x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-12-03 12:50:00 | 428.11 | 428.42 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:30:00 | 432.45 | 430.50 | 0.00 | ORB-long ORB[427.55,430.40] vol=1.9x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-12-04 10:50:00 | 431.47 | 431.13 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:05:00 | 441.85 | 443.37 | 0.00 | ORB-short ORB[443.00,446.30] vol=3.3x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-12-15 11:55:00 | 442.64 | 443.04 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:55:00 | 445.35 | 447.06 | 0.00 | ORB-short ORB[446.30,450.00] vol=2.0x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-12-23 11:25:00 | 446.22 | 446.71 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 11:15:00 | 446.30 | 444.95 | 0.00 | ORB-long ORB[441.70,446.20] vol=1.7x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-12-24 11:45:00 | 445.47 | 445.21 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:40:00 | 439.15 | 440.99 | 0.00 | ORB-short ORB[440.00,444.10] vol=2.0x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-12-26 10:50:00 | 440.10 | 440.79 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:05:00 | 435.60 | 438.37 | 0.00 | ORB-short ORB[437.05,442.15] vol=2.4x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-12-30 11:30:00 | 436.80 | 437.82 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-01-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:50:00 | 431.45 | 427.46 | 0.00 | ORB-long ORB[423.80,427.65] vol=3.1x ATR=1.69 |
| Stop hit — per-position SL triggered | 2026-01-08 09:55:00 | 429.76 | 427.95 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:55:00 | 420.50 | 417.86 | 0.00 | ORB-long ORB[414.10,419.00] vol=7.1x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 13:10:00 | 422.56 | 419.22 | 0.00 | T1 1.5R @ 422.56 |
| Stop hit — per-position SL triggered | 2026-01-14 13:40:00 | 420.50 | 419.70 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 408.20 | 407.09 | 0.00 | ORB-long ORB[404.05,407.00] vol=2.9x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-01-20 09:45:00 | 406.34 | 407.14 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:45:00 | 397.30 | 400.78 | 0.00 | ORB-short ORB[399.65,404.10] vol=2.3x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:55:00 | 394.97 | 399.92 | 0.00 | T1 1.5R @ 394.97 |
| Stop hit — per-position SL triggered | 2026-01-21 11:05:00 | 397.30 | 399.59 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 500.85 | 504.63 | 0.00 | ORB-short ORB[505.10,509.35] vol=2.2x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 498.69 | 504.32 | 0.00 | T1 1.5R @ 498.69 |
| Stop hit — per-position SL triggered | 2026-02-18 11:55:00 | 500.85 | 503.69 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:15:00 | 524.75 | 528.21 | 0.00 | ORB-short ORB[526.75,531.30] vol=4.8x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-03-16 10:25:00 | 526.75 | 527.80 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 557.55 | 553.63 | 0.00 | ORB-long ORB[549.15,555.90] vol=2.4x ATR=2.52 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 555.03 | 554.18 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:50:00 | 620.25 | 615.96 | 0.00 | ORB-long ORB[610.35,618.00] vol=3.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-05-07 12:50:00 | 617.32 | 617.03 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-05-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:10:00 | 585.20 | 593.72 | 0.00 | ORB-short ORB[592.30,600.00] vol=1.7x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:50:00 | 580.30 | 590.02 | 0.00 | T1 1.5R @ 580.30 |
| Target hit | 2026-05-08 15:05:00 | 580.00 | 578.59 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:35:00 | 483.25 | 2025-05-14 09:45:00 | 481.36 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-05-15 09:55:00 | 483.25 | 2025-05-15 10:15:00 | 481.31 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-05-28 10:05:00 | 494.85 | 2025-05-28 10:25:00 | 493.24 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-29 10:25:00 | 498.50 | 2025-05-29 10:50:00 | 499.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-30 09:30:00 | 509.70 | 2025-05-30 09:35:00 | 507.74 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-06-16 09:30:00 | 472.70 | 2025-06-16 10:30:00 | 474.61 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-17 09:30:00 | 478.30 | 2025-06-17 10:00:00 | 480.57 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-17 09:30:00 | 478.30 | 2025-06-17 12:00:00 | 480.00 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-19 10:35:00 | 463.15 | 2025-06-19 11:20:00 | 461.30 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-06-19 10:35:00 | 463.15 | 2025-06-19 15:20:00 | 455.10 | TARGET_HIT | 0.50 | 1.74% |
| BUY | retest1 | 2025-07-03 11:00:00 | 502.65 | 2025-07-03 11:15:00 | 505.41 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-07-03 11:00:00 | 502.65 | 2025-07-03 12:55:00 | 502.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 10:55:00 | 503.20 | 2025-07-04 11:05:00 | 501.43 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-11 09:35:00 | 509.30 | 2025-07-11 10:40:00 | 507.48 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-16 11:00:00 | 500.25 | 2025-07-16 11:05:00 | 501.47 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-17 10:50:00 | 504.15 | 2025-07-17 11:10:00 | 502.96 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-18 10:25:00 | 497.25 | 2025-07-18 10:40:00 | 498.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-18 10:45:00 | 406.70 | 2025-08-18 10:50:00 | 409.08 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-08-18 10:45:00 | 406.70 | 2025-08-18 10:55:00 | 406.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-21 11:10:00 | 425.00 | 2025-08-21 11:50:00 | 423.46 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-21 11:10:00 | 425.00 | 2025-08-21 15:20:00 | 420.60 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2025-08-22 09:30:00 | 417.55 | 2025-08-22 10:10:00 | 419.18 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-08-26 10:15:00 | 405.40 | 2025-08-26 11:05:00 | 402.94 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-08-26 10:15:00 | 405.40 | 2025-08-26 15:20:00 | 398.95 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2025-09-22 11:15:00 | 421.80 | 2025-09-22 11:45:00 | 423.03 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-23 09:30:00 | 418.30 | 2025-09-23 09:35:00 | 419.87 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-24 11:15:00 | 415.35 | 2025-09-24 11:40:00 | 416.29 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-30 09:40:00 | 414.30 | 2025-09-30 12:15:00 | 412.28 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-10-03 10:05:00 | 409.80 | 2025-10-03 10:25:00 | 407.90 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-03 10:05:00 | 409.80 | 2025-10-03 15:20:00 | 408.00 | TARGET_HIT | 0.50 | 0.44% |
| SELL | retest1 | 2025-10-06 10:55:00 | 405.50 | 2025-10-06 11:05:00 | 404.05 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-10-06 10:55:00 | 405.50 | 2025-10-06 12:35:00 | 405.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 10:05:00 | 404.85 | 2025-10-10 10:25:00 | 403.35 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-14 09:50:00 | 396.30 | 2025-10-14 10:15:00 | 394.65 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-10-14 09:50:00 | 396.30 | 2025-10-14 14:40:00 | 392.25 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2025-10-15 11:15:00 | 397.95 | 2025-10-15 11:30:00 | 399.54 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-15 11:15:00 | 397.95 | 2025-10-15 11:45:00 | 397.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-16 09:35:00 | 409.00 | 2025-10-16 09:45:00 | 407.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-17 11:10:00 | 405.90 | 2025-10-17 12:05:00 | 406.64 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-20 10:00:00 | 401.95 | 2025-10-20 10:45:00 | 400.42 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-20 10:00:00 | 401.95 | 2025-10-20 10:55:00 | 401.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 11:15:00 | 438.60 | 2025-10-28 11:55:00 | 436.86 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-28 11:15:00 | 438.60 | 2025-10-28 15:20:00 | 433.05 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2025-10-29 10:25:00 | 435.90 | 2025-10-29 10:30:00 | 438.34 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-10-29 10:25:00 | 435.90 | 2025-10-29 14:40:00 | 441.50 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2025-10-31 10:05:00 | 434.05 | 2025-10-31 10:10:00 | 435.13 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-03 10:25:00 | 444.50 | 2025-11-03 10:30:00 | 443.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-06 09:30:00 | 433.75 | 2025-11-06 10:10:00 | 431.77 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-11-06 09:30:00 | 433.75 | 2025-11-06 15:00:00 | 431.25 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-11-07 11:00:00 | 428.45 | 2025-11-07 11:30:00 | 429.92 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-11-07 11:00:00 | 428.45 | 2025-11-07 15:20:00 | 439.50 | TARGET_HIT | 0.50 | 2.58% |
| SELL | retest1 | 2025-11-11 11:10:00 | 438.85 | 2025-11-11 12:10:00 | 440.27 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-13 10:30:00 | 455.50 | 2025-11-13 10:35:00 | 453.81 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-17 09:30:00 | 450.90 | 2025-11-17 10:40:00 | 452.44 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-19 10:00:00 | 459.85 | 2025-11-19 10:05:00 | 461.90 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-19 10:00:00 | 459.85 | 2025-11-19 10:45:00 | 463.00 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-11-21 11:05:00 | 459.80 | 2025-11-21 14:10:00 | 457.85 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-21 11:05:00 | 459.80 | 2025-11-21 15:20:00 | 451.65 | TARGET_HIT | 0.50 | 1.77% |
| SELL | retest1 | 2025-11-24 11:10:00 | 443.35 | 2025-11-24 12:10:00 | 441.10 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-11-24 11:10:00 | 443.35 | 2025-11-24 15:20:00 | 434.80 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2025-11-28 09:45:00 | 427.25 | 2025-11-28 10:20:00 | 428.63 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-02 10:40:00 | 432.60 | 2025-12-02 11:20:00 | 431.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-03 10:25:00 | 426.85 | 2025-12-03 12:50:00 | 428.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-04 10:30:00 | 432.45 | 2025-12-04 10:50:00 | 431.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-15 11:05:00 | 441.85 | 2025-12-15 11:55:00 | 442.64 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-23 10:55:00 | 445.35 | 2025-12-23 11:25:00 | 446.22 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-24 11:15:00 | 446.30 | 2025-12-24 11:45:00 | 445.47 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-26 10:40:00 | 439.15 | 2025-12-26 10:50:00 | 440.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-30 11:05:00 | 435.60 | 2025-12-30 11:30:00 | 436.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-08 09:50:00 | 431.45 | 2026-01-08 09:55:00 | 429.76 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-01-14 10:55:00 | 420.50 | 2026-01-14 13:10:00 | 422.56 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-01-14 10:55:00 | 420.50 | 2026-01-14 13:40:00 | 420.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-20 09:35:00 | 408.20 | 2026-01-20 09:45:00 | 406.34 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-01-21 10:45:00 | 397.30 | 2026-01-21 10:55:00 | 394.97 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-01-21 10:45:00 | 397.30 | 2026-01-21 11:05:00 | 397.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 11:10:00 | 500.85 | 2026-02-18 11:25:00 | 498.69 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-18 11:10:00 | 500.85 | 2026-02-18 11:55:00 | 500.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:15:00 | 524.75 | 2026-03-16 10:25:00 | 526.75 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-10 09:35:00 | 557.55 | 2026-04-10 10:00:00 | 555.03 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-07 10:50:00 | 620.25 | 2026-05-07 12:50:00 | 617.32 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-05-08 10:10:00 | 585.20 | 2026-05-08 10:50:00 | 580.30 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2026-05-08 10:10:00 | 585.20 | 2026-05-08 15:05:00 | 580.00 | TARGET_HIT | 0.50 | 0.89% |
