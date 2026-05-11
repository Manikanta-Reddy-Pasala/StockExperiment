# Honasa Consumer Ltd. (HONASA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 358.30
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
| ENTRY1 | 40 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 35
- **Target hits / Stop hits / Partials:** 5 / 35 / 12
- **Avg / median % per leg:** 0.16% / -0.25%
- **Sum % (uncompounded):** 8.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 4 | 23.5% | 2 | 13 | 2 | 0.18% | 3.1% |
| BUY @ 2nd Alert (retest1) | 17 | 4 | 23.5% | 2 | 13 | 2 | 0.18% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 13 | 37.1% | 3 | 22 | 10 | 0.15% | 5.1% |
| SELL @ 2nd Alert (retest1) | 35 | 13 | 37.1% | 3 | 22 | 10 | 0.15% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 52 | 17 | 32.7% | 5 | 35 | 12 | 0.16% | 8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:30:00 | 425.10 | 426.66 | 0.00 | ORB-short ORB[426.00,429.50] vol=2.0x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 12:30:00 | 422.81 | 425.83 | 0.00 | T1 1.5R @ 422.81 |
| Stop hit — per-position SL triggered | 2024-05-13 13:00:00 | 425.10 | 425.72 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:40:00 | 420.30 | 422.23 | 0.00 | ORB-short ORB[422.00,425.95] vol=3.0x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-05-14 10:25:00 | 421.37 | 421.78 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:50:00 | 413.30 | 415.11 | 0.00 | ORB-short ORB[414.10,416.95] vol=2.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-05-16 11:05:00 | 414.52 | 415.06 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:55:00 | 431.30 | 428.73 | 0.00 | ORB-long ORB[424.45,430.75] vol=1.5x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-05-22 10:05:00 | 429.20 | 428.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 10:40:00 | 430.85 | 428.48 | 0.00 | ORB-long ORB[424.25,428.95] vol=7.7x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 429.38 | 428.60 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 10:55:00 | 434.40 | 440.07 | 0.00 | ORB-short ORB[437.55,443.60] vol=3.0x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:00:00 | 430.82 | 439.50 | 0.00 | T1 1.5R @ 430.82 |
| Stop hit — per-position SL triggered | 2024-06-11 12:40:00 | 434.40 | 436.64 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 11:15:00 | 431.05 | 433.56 | 0.00 | ORB-short ORB[434.00,436.60] vol=2.3x ATR=1.24 |
| Stop hit — per-position SL triggered | 2024-06-14 11:30:00 | 432.29 | 433.47 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:40:00 | 431.15 | 428.30 | 0.00 | ORB-long ORB[424.70,429.80] vol=1.6x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:45:00 | 433.72 | 431.01 | 0.00 | T1 1.5R @ 433.72 |
| Target hit | 2024-06-21 14:50:00 | 443.55 | 444.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2024-06-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:55:00 | 451.30 | 452.88 | 0.00 | ORB-short ORB[452.30,457.00] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-06-25 11:10:00 | 452.93 | 452.86 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 439.55 | 442.36 | 0.00 | ORB-short ORB[442.10,446.95] vol=1.7x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:05:00 | 436.89 | 440.27 | 0.00 | T1 1.5R @ 436.89 |
| Target hit | 2024-06-27 11:00:00 | 434.05 | 433.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2024-07-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:05:00 | 448.00 | 442.33 | 0.00 | ORB-long ORB[432.00,437.45] vol=1.9x ATR=2.36 |
| Stop hit — per-position SL triggered | 2024-07-01 10:35:00 | 445.64 | 444.32 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:55:00 | 438.00 | 441.80 | 0.00 | ORB-short ORB[441.30,446.00] vol=2.9x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-07-02 11:20:00 | 439.44 | 441.41 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:45:00 | 475.05 | 471.31 | 0.00 | ORB-long ORB[467.10,471.35] vol=2.1x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-07-12 11:05:00 | 473.17 | 471.82 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:00:00 | 464.70 | 463.19 | 0.00 | ORB-long ORB[460.00,463.85] vol=5.9x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 463.32 | 463.53 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:15:00 | 468.40 | 465.89 | 0.00 | ORB-long ORB[462.15,465.70] vol=2.0x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-07-30 10:30:00 | 466.68 | 466.11 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:15:00 | 465.30 | 467.66 | 0.00 | ORB-short ORB[466.40,469.75] vol=3.0x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-08-01 11:20:00 | 466.24 | 467.56 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:35:00 | 451.00 | 454.03 | 0.00 | ORB-short ORB[452.00,458.05] vol=1.7x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-08-02 10:05:00 | 453.50 | 452.34 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 454.25 | 449.29 | 0.00 | ORB-long ORB[449.05,453.70] vol=5.1x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 452.29 | 450.24 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 458.90 | 456.59 | 0.00 | ORB-long ORB[451.55,456.15] vol=3.4x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:45:00 | 461.25 | 457.17 | 0.00 | T1 1.5R @ 461.25 |
| Target hit | 2024-08-08 13:35:00 | 480.00 | 480.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2024-08-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:35:00 | 469.15 | 466.20 | 0.00 | ORB-long ORB[461.35,468.00] vol=3.0x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-08-19 12:20:00 | 467.28 | 467.76 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:25:00 | 465.80 | 466.97 | 0.00 | ORB-short ORB[465.90,471.55] vol=3.7x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-08-21 10:45:00 | 467.21 | 466.95 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:40:00 | 464.60 | 466.49 | 0.00 | ORB-short ORB[466.45,470.45] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 466.00 | 466.65 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 508.70 | 512.00 | 0.00 | ORB-short ORB[510.60,518.25] vol=1.5x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 09:45:00 | 504.24 | 510.40 | 0.00 | T1 1.5R @ 504.24 |
| Stop hit — per-position SL triggered | 2024-08-30 09:55:00 | 508.70 | 509.81 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:10:00 | 425.20 | 426.21 | 0.00 | ORB-short ORB[428.60,432.00] vol=4.7x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-10-14 11:45:00 | 426.43 | 426.14 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:50:00 | 419.85 | 422.43 | 0.00 | ORB-short ORB[423.95,430.15] vol=10.7x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-10-16 09:55:00 | 421.44 | 422.27 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:10:00 | 411.70 | 415.01 | 0.00 | ORB-short ORB[413.70,419.00] vol=2.5x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:20:00 | 409.62 | 414.18 | 0.00 | T1 1.5R @ 409.62 |
| Stop hit — per-position SL triggered | 2024-10-17 12:00:00 | 411.70 | 413.62 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 09:50:00 | 403.30 | 404.77 | 0.00 | ORB-short ORB[403.50,409.20] vol=1.8x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:15:00 | 400.76 | 404.14 | 0.00 | T1 1.5R @ 400.76 |
| Stop hit — per-position SL triggered | 2024-10-31 11:40:00 | 403.30 | 402.69 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-11-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 09:35:00 | 400.85 | 396.81 | 0.00 | ORB-long ORB[392.45,396.90] vol=1.8x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-11-04 11:10:00 | 399.03 | 398.79 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 252.00 | 253.07 | 0.00 | ORB-short ORB[252.50,254.80] vol=1.8x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-12-24 09:40:00 | 253.05 | 252.98 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:15:00 | 257.00 | 254.48 | 0.00 | ORB-long ORB[252.25,256.00] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-12-26 10:25:00 | 255.46 | 254.70 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-12-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:05:00 | 248.55 | 250.99 | 0.00 | ORB-short ORB[249.95,253.40] vol=1.8x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:50:00 | 247.19 | 250.12 | 0.00 | T1 1.5R @ 247.19 |
| Stop hit — per-position SL triggered | 2024-12-27 13:55:00 | 248.55 | 249.63 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-01-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:30:00 | 251.20 | 252.85 | 0.00 | ORB-short ORB[253.60,256.45] vol=2.2x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 13:05:00 | 249.71 | 251.31 | 0.00 | T1 1.5R @ 249.71 |
| Target hit | 2025-01-01 15:20:00 | 249.60 | 250.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-01-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:00:00 | 248.60 | 249.53 | 0.00 | ORB-short ORB[249.10,252.45] vol=2.2x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-01-02 11:05:00 | 249.26 | 249.52 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-01-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:40:00 | 247.80 | 246.64 | 0.00 | ORB-long ORB[244.15,247.00] vol=2.0x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-01-07 09:45:00 | 246.85 | 246.72 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:35:00 | 241.70 | 243.71 | 0.00 | ORB-short ORB[243.15,246.20] vol=2.0x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-01-10 09:40:00 | 242.75 | 243.63 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:35:00 | 248.40 | 247.34 | 0.00 | ORB-long ORB[245.15,248.00] vol=5.2x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-01-17 10:40:00 | 247.70 | 247.35 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-02-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 10:50:00 | 227.99 | 229.96 | 0.00 | ORB-short ORB[229.55,232.98] vol=3.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 11:05:00 | 226.72 | 229.43 | 0.00 | T1 1.5R @ 226.72 |
| Target hit | 2025-02-05 15:20:00 | 223.74 | 226.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 225.94 | 227.50 | 0.00 | ORB-short ORB[226.87,229.80] vol=3.1x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-02-21 09:45:00 | 227.00 | 227.37 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-04-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:05:00 | 238.19 | 235.35 | 0.00 | ORB-long ORB[232.63,235.99] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-04-02 10:10:00 | 237.01 | 235.50 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:40:00 | 229.37 | 231.72 | 0.00 | ORB-short ORB[232.60,235.70] vol=1.8x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:55:00 | 227.54 | 231.41 | 0.00 | T1 1.5R @ 227.54 |
| Stop hit — per-position SL triggered | 2025-04-23 11:00:00 | 229.37 | 231.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:30:00 | 425.10 | 2024-05-13 12:30:00 | 422.81 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-05-13 10:30:00 | 425.10 | 2024-05-13 13:00:00 | 425.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-14 09:40:00 | 420.30 | 2024-05-14 10:25:00 | 421.37 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-16 10:50:00 | 413.30 | 2024-05-16 11:05:00 | 414.52 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-22 09:55:00 | 431.30 | 2024-05-22 10:05:00 | 429.20 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-05-31 10:40:00 | 430.85 | 2024-05-31 10:45:00 | 429.38 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-11 10:55:00 | 434.40 | 2024-06-11 11:00:00 | 430.82 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2024-06-11 10:55:00 | 434.40 | 2024-06-11 12:40:00 | 434.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 11:15:00 | 431.05 | 2024-06-14 11:30:00 | 432.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-21 09:40:00 | 431.15 | 2024-06-21 09:45:00 | 433.72 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-06-21 09:40:00 | 431.15 | 2024-06-21 14:50:00 | 443.55 | TARGET_HIT | 0.50 | 2.88% |
| SELL | retest1 | 2024-06-25 10:55:00 | 451.30 | 2024-06-25 11:10:00 | 452.93 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-27 09:40:00 | 439.55 | 2024-06-27 10:05:00 | 436.89 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-06-27 09:40:00 | 439.55 | 2024-06-27 11:00:00 | 434.05 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2024-07-01 10:05:00 | 448.00 | 2024-07-01 10:35:00 | 445.64 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-07-02 10:55:00 | 438.00 | 2024-07-02 11:20:00 | 439.44 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-12 10:45:00 | 475.05 | 2024-07-12 11:05:00 | 473.17 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-25 10:00:00 | 464.70 | 2024-07-25 11:15:00 | 463.32 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-30 10:15:00 | 468.40 | 2024-07-30 10:30:00 | 466.68 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-01 11:15:00 | 465.30 | 2024-08-01 11:20:00 | 466.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-02 09:35:00 | 451.00 | 2024-08-02 10:05:00 | 453.50 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-08-07 10:45:00 | 454.25 | 2024-08-07 11:15:00 | 452.29 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-08-08 11:15:00 | 458.90 | 2024-08-08 11:45:00 | 461.25 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-08 11:15:00 | 458.90 | 2024-08-08 13:35:00 | 480.00 | TARGET_HIT | 0.50 | 4.60% |
| BUY | retest1 | 2024-08-19 10:35:00 | 469.15 | 2024-08-19 12:20:00 | 467.28 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-21 10:25:00 | 465.80 | 2024-08-21 10:45:00 | 467.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-22 10:40:00 | 464.60 | 2024-08-22 11:15:00 | 466.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-30 09:35:00 | 508.70 | 2024-08-30 09:45:00 | 504.24 | PARTIAL | 0.50 | 0.88% |
| SELL | retest1 | 2024-08-30 09:35:00 | 508.70 | 2024-08-30 09:55:00 | 508.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-14 11:10:00 | 425.20 | 2024-10-14 11:45:00 | 426.43 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-16 09:50:00 | 419.85 | 2024-10-16 09:55:00 | 421.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-17 11:10:00 | 411.70 | 2024-10-17 11:20:00 | 409.62 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-17 11:10:00 | 411.70 | 2024-10-17 12:00:00 | 411.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-31 09:50:00 | 403.30 | 2024-10-31 10:15:00 | 400.76 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-31 09:50:00 | 403.30 | 2024-10-31 11:40:00 | 403.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-04 09:35:00 | 400.85 | 2024-11-04 11:10:00 | 399.03 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-12-24 09:30:00 | 252.00 | 2024-12-24 09:40:00 | 253.05 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-26 10:15:00 | 257.00 | 2024-12-26 10:25:00 | 255.46 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-12-27 10:05:00 | 248.55 | 2024-12-27 11:50:00 | 247.19 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-12-27 10:05:00 | 248.55 | 2024-12-27 13:55:00 | 248.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 10:30:00 | 251.20 | 2025-01-01 13:05:00 | 249.71 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-01 10:30:00 | 251.20 | 2025-01-01 15:20:00 | 249.60 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2025-01-02 11:00:00 | 248.60 | 2025-01-02 11:05:00 | 249.26 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-07 09:40:00 | 247.80 | 2025-01-07 09:45:00 | 246.85 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-10 09:35:00 | 241.70 | 2025-01-10 09:40:00 | 242.75 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-17 10:35:00 | 248.40 | 2025-01-17 10:40:00 | 247.70 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-05 10:50:00 | 227.99 | 2025-02-05 11:05:00 | 226.72 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-05 10:50:00 | 227.99 | 2025-02-05 15:20:00 | 223.74 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2025-02-21 09:40:00 | 225.94 | 2025-02-21 09:45:00 | 227.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-04-02 10:05:00 | 238.19 | 2025-04-02 10:10:00 | 237.01 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-04-23 10:40:00 | 229.37 | 2025-04-23 10:55:00 | 227.54 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2025-04-23 10:40:00 | 229.37 | 2025-04-23 11:00:00 | 229.37 | STOP_HIT | 0.50 | 0.00% |
