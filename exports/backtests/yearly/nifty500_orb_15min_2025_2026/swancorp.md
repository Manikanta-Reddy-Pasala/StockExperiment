# Swan Corp Ltd. (SWANCORP)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 353.15
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
| ENTRY1 | 69 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 10 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 59
- **Target hits / Stop hits / Partials:** 10 / 59 / 24
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 13.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 20 | 35.1% | 7 | 37 | 13 | 0.17% | 9.9% |
| BUY @ 2nd Alert (retest1) | 57 | 20 | 35.1% | 7 | 37 | 13 | 0.17% | 9.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 14 | 38.9% | 3 | 22 | 11 | 0.11% | 3.8% |
| SELL @ 2nd Alert (retest1) | 36 | 14 | 38.9% | 3 | 22 | 11 | 0.11% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 93 | 34 | 36.6% | 10 | 59 | 24 | 0.15% | 13.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:50:00 | 449.80 | 446.97 | 0.00 | ORB-long ORB[441.55,446.45] vol=6.0x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-05-28 10:00:00 | 447.83 | 447.34 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:55:00 | 419.90 | 415.07 | 0.00 | ORB-long ORB[412.85,417.40] vol=3.4x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:10:00 | 422.82 | 417.06 | 0.00 | T1 1.5R @ 422.82 |
| Target hit | 2025-06-04 14:55:00 | 447.40 | 447.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2025-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:35:00 | 460.60 | 457.36 | 0.00 | ORB-long ORB[452.90,459.35] vol=2.5x ATR=2.62 |
| Stop hit — per-position SL triggered | 2025-06-09 10:00:00 | 457.98 | 458.17 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:10:00 | 438.30 | 434.65 | 0.00 | ORB-long ORB[431.30,437.00] vol=1.6x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-06-13 12:10:00 | 435.74 | 435.87 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 427.90 | 424.61 | 0.00 | ORB-long ORB[421.20,426.85] vol=2.3x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:35:00 | 430.64 | 427.25 | 0.00 | T1 1.5R @ 430.64 |
| Target hit | 2025-06-19 09:50:00 | 428.70 | 429.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2025-06-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:30:00 | 448.45 | 444.75 | 0.00 | ORB-long ORB[441.00,447.60] vol=2.3x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:35:00 | 452.00 | 445.48 | 0.00 | T1 1.5R @ 452.00 |
| Stop hit — per-position SL triggered | 2025-06-25 10:40:00 | 448.45 | 445.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:00:00 | 455.40 | 451.75 | 0.00 | ORB-long ORB[449.20,454.00] vol=2.2x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-06-30 10:10:00 | 453.34 | 452.72 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 439.00 | 442.77 | 0.00 | ORB-short ORB[442.05,445.90] vol=1.8x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:45:00 | 436.67 | 442.09 | 0.00 | T1 1.5R @ 436.67 |
| Stop hit — per-position SL triggered | 2025-07-02 11:25:00 | 439.00 | 439.59 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:40:00 | 442.30 | 439.67 | 0.00 | ORB-long ORB[436.50,442.15] vol=2.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-07-03 09:50:00 | 440.52 | 439.85 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:15:00 | 445.55 | 442.41 | 0.00 | ORB-long ORB[438.55,443.90] vol=1.9x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:20:00 | 447.69 | 447.30 | 0.00 | T1 1.5R @ 447.69 |
| Target hit | 2025-07-09 11:05:00 | 453.80 | 453.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2025-07-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:00:00 | 457.10 | 452.55 | 0.00 | ORB-long ORB[448.00,453.50] vol=2.7x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-07-14 10:05:00 | 455.04 | 452.72 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-08-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:55:00 | 459.60 | 464.68 | 0.00 | ORB-short ORB[462.85,468.20] vol=2.3x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:45:00 | 456.60 | 462.43 | 0.00 | T1 1.5R @ 456.60 |
| Stop hit — per-position SL triggered | 2025-08-01 11:05:00 | 459.60 | 462.16 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-08-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:30:00 | 423.60 | 418.14 | 0.00 | ORB-long ORB[412.20,418.35] vol=4.3x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-08-11 09:45:00 | 421.04 | 419.73 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-08-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:30:00 | 429.50 | 426.77 | 0.00 | ORB-long ORB[423.15,428.75] vol=2.9x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:50:00 | 431.88 | 428.21 | 0.00 | T1 1.5R @ 431.88 |
| Target hit | 2025-08-13 15:20:00 | 441.70 | 432.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-08-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 11:05:00 | 472.45 | 464.77 | 0.00 | ORB-long ORB[457.10,464.00] vol=13.5x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:15:00 | 475.85 | 471.95 | 0.00 | T1 1.5R @ 475.85 |
| Target hit | 2025-08-22 11:55:00 | 483.95 | 484.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2025-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:35:00 | 465.80 | 462.40 | 0.00 | ORB-long ORB[458.20,465.00] vol=1.8x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-09-03 09:55:00 | 463.62 | 463.60 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-09-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:45:00 | 447.40 | 452.24 | 0.00 | ORB-short ORB[450.85,455.90] vol=2.9x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-09-09 11:05:00 | 449.31 | 451.36 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 471.00 | 466.87 | 0.00 | ORB-long ORB[460.30,467.30] vol=3.3x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-09-17 10:35:00 | 468.24 | 468.63 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:00:00 | 480.40 | 486.02 | 0.00 | ORB-short ORB[485.30,492.50] vol=2.2x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-09-23 11:45:00 | 482.14 | 485.25 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:30:00 | 496.45 | 493.18 | 0.00 | ORB-long ORB[486.40,493.45] vol=3.5x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-09-25 09:35:00 | 493.97 | 493.28 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:45:00 | 482.30 | 476.94 | 0.00 | ORB-long ORB[474.40,478.75] vol=1.7x ATR=2.26 |
| Stop hit — per-position SL triggered | 2025-09-26 09:50:00 | 480.04 | 477.13 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-10-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:40:00 | 462.25 | 459.26 | 0.00 | ORB-long ORB[455.30,459.90] vol=2.3x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:45:00 | 464.82 | 460.28 | 0.00 | T1 1.5R @ 464.82 |
| Stop hit — per-position SL triggered | 2025-10-01 09:50:00 | 462.25 | 460.26 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 451.85 | 457.89 | 0.00 | ORB-short ORB[457.60,462.40] vol=2.4x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-10-06 11:30:00 | 453.29 | 457.66 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:40:00 | 450.70 | 453.08 | 0.00 | ORB-short ORB[453.40,456.60] vol=2.4x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:00:00 | 449.05 | 452.45 | 0.00 | T1 1.5R @ 449.05 |
| Stop hit — per-position SL triggered | 2025-10-07 12:30:00 | 450.70 | 451.56 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:10:00 | 446.40 | 448.78 | 0.00 | ORB-short ORB[447.75,451.00] vol=1.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-10-08 10:25:00 | 447.69 | 448.48 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 444.60 | 442.15 | 0.00 | ORB-long ORB[437.80,443.50] vol=2.3x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-10-10 09:45:00 | 442.96 | 442.25 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:35:00 | 435.40 | 433.50 | 0.00 | ORB-long ORB[430.55,433.90] vol=1.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-10-16 09:45:00 | 434.20 | 433.81 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 09:45:00 | 426.85 | 428.54 | 0.00 | ORB-short ORB[428.05,431.35] vol=1.6x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:00:00 | 425.01 | 427.94 | 0.00 | T1 1.5R @ 425.01 |
| Stop hit — per-position SL triggered | 2025-10-20 10:10:00 | 426.85 | 427.60 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:55:00 | 431.50 | 430.43 | 0.00 | ORB-long ORB[428.20,431.00] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-10-24 10:05:00 | 430.60 | 430.53 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:45:00 | 435.60 | 433.82 | 0.00 | ORB-long ORB[430.90,433.90] vol=3.0x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-10-27 10:00:00 | 434.20 | 433.92 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:30:00 | 469.00 | 471.59 | 0.00 | ORB-short ORB[469.20,474.90] vol=3.0x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:45:00 | 466.63 | 470.46 | 0.00 | T1 1.5R @ 466.63 |
| Target hit | 2025-11-04 15:20:00 | 454.10 | 462.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-11-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:35:00 | 472.25 | 469.18 | 0.00 | ORB-long ORB[467.10,471.55] vol=3.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-11-12 10:40:00 | 470.50 | 469.63 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-11-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:40:00 | 470.40 | 467.78 | 0.00 | ORB-long ORB[464.15,469.00] vol=2.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-11-13 11:05:00 | 469.10 | 468.10 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-11-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:10:00 | 474.40 | 469.77 | 0.00 | ORB-long ORB[466.10,469.90] vol=3.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 472.45 | 470.00 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 445.00 | 447.03 | 0.00 | ORB-short ORB[445.50,449.45] vol=2.0x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:40:00 | 442.59 | 445.89 | 0.00 | T1 1.5R @ 442.59 |
| Target hit | 2025-11-26 11:15:00 | 444.45 | 444.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — BUY (started 2025-11-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:45:00 | 447.80 | 442.35 | 0.00 | ORB-long ORB[437.25,442.35] vol=6.8x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-11-28 11:00:00 | 445.87 | 443.51 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-12-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 11:05:00 | 454.35 | 452.03 | 0.00 | ORB-long ORB[447.45,450.95] vol=2.5x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-12-02 11:20:00 | 452.66 | 452.17 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 09:35:00 | 453.00 | 449.79 | 0.00 | ORB-long ORB[446.25,451.00] vol=3.5x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-12-03 09:40:00 | 451.32 | 450.01 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 10:10:00 | 444.75 | 446.36 | 0.00 | ORB-short ORB[445.05,449.00] vol=2.1x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 446.14 | 446.35 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-12-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 11:00:00 | 448.50 | 448.81 | 0.00 | ORB-short ORB[448.80,453.70] vol=2.1x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:50:00 | 446.59 | 448.59 | 0.00 | T1 1.5R @ 446.59 |
| Target hit | 2025-12-05 15:20:00 | 446.80 | 447.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2025-12-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:35:00 | 440.10 | 442.79 | 0.00 | ORB-short ORB[441.85,446.80] vol=1.8x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:00:00 | 437.90 | 440.82 | 0.00 | T1 1.5R @ 437.90 |
| Stop hit — per-position SL triggered | 2025-12-08 11:10:00 | 440.10 | 440.35 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:15:00 | 445.00 | 439.45 | 0.00 | ORB-long ORB[435.95,440.65] vol=2.9x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 11:35:00 | 448.05 | 440.88 | 0.00 | T1 1.5R @ 448.05 |
| Stop hit — per-position SL triggered | 2025-12-09 13:30:00 | 445.00 | 443.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:20:00 | 452.95 | 450.11 | 0.00 | ORB-long ORB[447.95,452.75] vol=2.7x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-12-11 10:30:00 | 451.08 | 450.14 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 461.40 | 459.38 | 0.00 | ORB-long ORB[456.65,460.40] vol=2.2x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-12-12 09:40:00 | 459.56 | 459.78 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:30:00 | 452.30 | 454.87 | 0.00 | ORB-short ORB[453.00,456.00] vol=3.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-12-15 10:35:00 | 454.25 | 454.79 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:40:00 | 455.25 | 456.52 | 0.00 | ORB-short ORB[456.30,461.05] vol=2.0x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:30:00 | 453.59 | 456.07 | 0.00 | T1 1.5R @ 453.59 |
| Stop hit — per-position SL triggered | 2025-12-16 13:55:00 | 455.25 | 455.36 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 10:00:00 | 462.30 | 459.61 | 0.00 | ORB-long ORB[457.10,460.15] vol=1.8x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 11:15:00 | 465.25 | 462.09 | 0.00 | T1 1.5R @ 465.25 |
| Target hit | 2025-12-17 12:00:00 | 462.65 | 463.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2025-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 09:30:00 | 476.15 | 471.89 | 0.00 | ORB-long ORB[468.20,473.70] vol=1.9x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-12-18 09:45:00 | 473.15 | 473.44 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 09:40:00 | 468.05 | 469.08 | 0.00 | ORB-short ORB[468.10,471.00] vol=1.7x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-12-22 09:50:00 | 469.64 | 469.56 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 465.50 | 462.09 | 0.00 | ORB-long ORB[459.00,464.10] vol=4.4x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:20:00 | 467.81 | 463.41 | 0.00 | T1 1.5R @ 467.81 |
| Stop hit — per-position SL triggered | 2025-12-26 11:45:00 | 465.50 | 463.83 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:40:00 | 463.50 | 466.14 | 0.00 | ORB-short ORB[465.55,468.25] vol=1.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-01-02 10:45:00 | 464.83 | 465.57 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-01-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:45:00 | 466.30 | 463.39 | 0.00 | ORB-long ORB[460.50,465.25] vol=1.9x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:55:00 | 468.95 | 467.11 | 0.00 | T1 1.5R @ 468.95 |
| Target hit | 2026-01-05 12:45:00 | 471.00 | 472.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2026-01-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:55:00 | 466.50 | 467.36 | 0.00 | ORB-short ORB[466.60,469.95] vol=1.6x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:10:00 | 464.31 | 467.01 | 0.00 | T1 1.5R @ 464.31 |
| Stop hit — per-position SL triggered | 2026-01-07 10:55:00 | 466.50 | 466.13 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:55:00 | 470.35 | 466.62 | 0.00 | ORB-long ORB[462.40,468.20] vol=1.5x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-01-08 10:05:00 | 468.64 | 466.95 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:45:00 | 425.25 | 421.09 | 0.00 | ORB-long ORB[418.00,421.65] vol=2.1x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-02-01 10:55:00 | 423.69 | 421.37 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:45:00 | 419.50 | 423.04 | 0.00 | ORB-short ORB[420.00,425.00] vol=1.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 420.94 | 422.85 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 412.35 | 414.45 | 0.00 | ORB-short ORB[413.35,417.35] vol=1.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-02-11 10:45:00 | 413.71 | 413.46 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 408.15 | 411.78 | 0.00 | ORB-short ORB[410.50,415.50] vol=1.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 409.58 | 411.13 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:10:00 | 409.00 | 407.57 | 0.00 | ORB-long ORB[405.30,408.70] vol=1.5x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-16 10:25:00 | 407.89 | 407.67 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 412.40 | 409.20 | 0.00 | ORB-long ORB[405.50,410.90] vol=1.9x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 411.18 | 409.35 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 406.20 | 407.93 | 0.00 | ORB-short ORB[408.20,411.30] vol=1.9x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:50:00 | 404.97 | 407.26 | 0.00 | T1 1.5R @ 404.97 |
| Stop hit — per-position SL triggered | 2026-02-19 13:05:00 | 406.20 | 406.57 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 400.95 | 404.49 | 0.00 | ORB-short ORB[405.20,410.45] vol=1.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 402.37 | 404.04 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 347.45 | 353.01 | 0.00 | ORB-short ORB[356.40,361.50] vol=2.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 349.12 | 352.89 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 352.10 | 349.14 | 0.00 | ORB-long ORB[347.10,349.90] vol=1.6x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 354.24 | 350.03 | 0.00 | T1 1.5R @ 354.24 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 352.10 | 350.12 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:50:00 | 353.20 | 348.21 | 0.00 | ORB-long ORB[344.50,348.50] vol=3.1x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-04-23 09:55:00 | 351.38 | 349.35 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 335.75 | 333.44 | 0.00 | ORB-long ORB[330.75,334.70] vol=2.0x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 338.02 | 335.12 | 0.00 | T1 1.5R @ 338.02 |
| Stop hit — per-position SL triggered | 2026-04-27 13:00:00 | 335.75 | 336.74 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:15:00 | 342.20 | 339.19 | 0.00 | ORB-long ORB[336.60,340.85] vol=1.8x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-04-28 10:25:00 | 340.93 | 339.39 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 350.90 | 346.94 | 0.00 | ORB-long ORB[342.65,345.75] vol=1.9x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 348.69 | 348.84 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 352.00 | 353.76 | 0.00 | ORB-short ORB[353.05,356.00] vol=2.0x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 353.22 | 353.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-28 09:50:00 | 449.80 | 2025-05-28 10:00:00 | 447.83 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-06-04 10:55:00 | 419.90 | 2025-06-04 11:10:00 | 422.82 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-06-04 10:55:00 | 419.90 | 2025-06-04 14:55:00 | 447.40 | TARGET_HIT | 0.50 | 6.55% |
| BUY | retest1 | 2025-06-09 09:35:00 | 460.60 | 2025-06-09 10:00:00 | 457.98 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-06-13 10:10:00 | 438.30 | 2025-06-13 12:10:00 | 435.74 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2025-06-19 09:30:00 | 427.90 | 2025-06-19 09:35:00 | 430.64 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-06-19 09:30:00 | 427.90 | 2025-06-19 09:50:00 | 428.70 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-06-25 10:30:00 | 448.45 | 2025-06-25 10:35:00 | 452.00 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-06-25 10:30:00 | 448.45 | 2025-06-25 10:40:00 | 448.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-30 10:00:00 | 455.40 | 2025-06-30 10:10:00 | 453.34 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-07-02 09:40:00 | 439.00 | 2025-07-02 09:45:00 | 436.67 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-07-02 09:40:00 | 439.00 | 2025-07-02 11:25:00 | 439.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 09:40:00 | 442.30 | 2025-07-03 09:50:00 | 440.52 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-07-09 10:15:00 | 445.55 | 2025-07-09 10:20:00 | 447.69 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-09 10:15:00 | 445.55 | 2025-07-09 11:05:00 | 453.80 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2025-07-14 10:00:00 | 457.10 | 2025-07-14 10:05:00 | 455.04 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-08-01 09:55:00 | 459.60 | 2025-08-01 10:45:00 | 456.60 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-08-01 09:55:00 | 459.60 | 2025-08-01 11:05:00 | 459.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 09:30:00 | 423.60 | 2025-08-11 09:45:00 | 421.04 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2025-08-13 10:30:00 | 429.50 | 2025-08-13 10:50:00 | 431.88 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-08-13 10:30:00 | 429.50 | 2025-08-13 15:20:00 | 441.70 | TARGET_HIT | 0.50 | 2.84% |
| BUY | retest1 | 2025-08-22 11:05:00 | 472.45 | 2025-08-22 11:15:00 | 475.85 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-08-22 11:05:00 | 472.45 | 2025-08-22 11:55:00 | 483.95 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2025-09-03 09:35:00 | 465.80 | 2025-09-03 09:55:00 | 463.62 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-09-09 10:45:00 | 447.40 | 2025-09-09 11:05:00 | 449.31 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-09-17 09:35:00 | 471.00 | 2025-09-17 10:35:00 | 468.24 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2025-09-23 11:00:00 | 480.40 | 2025-09-23 11:45:00 | 482.14 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-25 09:30:00 | 496.45 | 2025-09-25 09:35:00 | 493.97 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-09-26 09:45:00 | 482.30 | 2025-09-26 09:50:00 | 480.04 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-10-01 09:40:00 | 462.25 | 2025-10-01 09:45:00 | 464.82 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-10-01 09:40:00 | 462.25 | 2025-10-01 09:50:00 | 462.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 11:10:00 | 451.85 | 2025-10-06 11:30:00 | 453.29 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-07 10:40:00 | 450.70 | 2025-10-07 11:00:00 | 449.05 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-07 10:40:00 | 450.70 | 2025-10-07 12:30:00 | 450.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 10:10:00 | 446.40 | 2025-10-08 10:25:00 | 447.69 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-10 09:40:00 | 444.60 | 2025-10-10 09:45:00 | 442.96 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-16 09:35:00 | 435.40 | 2025-10-16 09:45:00 | 434.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-20 09:45:00 | 426.85 | 2025-10-20 10:00:00 | 425.01 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-20 09:45:00 | 426.85 | 2025-10-20 10:10:00 | 426.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-24 09:55:00 | 431.50 | 2025-10-24 10:05:00 | 430.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-27 09:45:00 | 435.60 | 2025-10-27 10:00:00 | 434.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-04 09:30:00 | 469.00 | 2025-11-04 09:45:00 | 466.63 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-11-04 09:30:00 | 469.00 | 2025-11-04 15:20:00 | 454.10 | TARGET_HIT | 0.50 | 3.18% |
| BUY | retest1 | 2025-11-12 10:35:00 | 472.25 | 2025-11-12 10:40:00 | 470.50 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-13 10:40:00 | 470.40 | 2025-11-13 11:05:00 | 469.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-20 10:10:00 | 474.40 | 2025-11-20 10:15:00 | 472.45 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-11-26 09:30:00 | 445.00 | 2025-11-26 09:40:00 | 442.59 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-11-26 09:30:00 | 445.00 | 2025-11-26 11:15:00 | 444.45 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2025-11-28 10:45:00 | 447.80 | 2025-11-28 11:00:00 | 445.87 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-12-02 11:05:00 | 454.35 | 2025-12-02 11:20:00 | 452.66 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-03 09:35:00 | 453.00 | 2025-12-03 09:40:00 | 451.32 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-12-04 10:10:00 | 444.75 | 2025-12-04 10:15:00 | 446.14 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-05 11:00:00 | 448.50 | 2025-12-05 11:50:00 | 446.59 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-05 11:00:00 | 448.50 | 2025-12-05 15:20:00 | 446.80 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-08 09:35:00 | 440.10 | 2025-12-08 10:00:00 | 437.90 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-12-08 09:35:00 | 440.10 | 2025-12-08 11:10:00 | 440.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-09 11:15:00 | 445.00 | 2025-12-09 11:35:00 | 448.05 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-12-09 11:15:00 | 445.00 | 2025-12-09 13:30:00 | 445.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:20:00 | 452.95 | 2025-12-11 10:30:00 | 451.08 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-12-12 09:30:00 | 461.40 | 2025-12-12 09:40:00 | 459.56 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-12-15 10:30:00 | 452.30 | 2025-12-15 10:35:00 | 454.25 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-12-16 10:40:00 | 455.25 | 2025-12-16 11:30:00 | 453.59 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-16 10:40:00 | 455.25 | 2025-12-16 13:55:00 | 455.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 10:00:00 | 462.30 | 2025-12-17 11:15:00 | 465.25 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-12-17 10:00:00 | 462.30 | 2025-12-17 12:00:00 | 462.65 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2025-12-18 09:30:00 | 476.15 | 2025-12-18 09:45:00 | 473.15 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2025-12-22 09:40:00 | 468.05 | 2025-12-22 09:50:00 | 469.64 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-26 11:10:00 | 465.50 | 2025-12-26 11:20:00 | 467.81 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-12-26 11:10:00 | 465.50 | 2025-12-26 11:45:00 | 465.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-02 10:40:00 | 463.50 | 2026-01-02 10:45:00 | 464.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-05 09:45:00 | 466.30 | 2026-01-05 09:55:00 | 468.95 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-01-05 09:45:00 | 466.30 | 2026-01-05 12:45:00 | 471.00 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-01-07 09:55:00 | 466.50 | 2026-01-07 10:10:00 | 464.31 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-01-07 09:55:00 | 466.50 | 2026-01-07 10:55:00 | 466.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-08 09:55:00 | 470.35 | 2026-01-08 10:05:00 | 468.64 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-01 10:45:00 | 425.25 | 2026-02-01 10:55:00 | 423.69 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-10 10:45:00 | 419.50 | 2026-02-10 11:00:00 | 420.94 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-11 09:35:00 | 412.35 | 2026-02-11 10:45:00 | 413.71 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-13 09:30:00 | 408.15 | 2026-02-13 09:40:00 | 409.58 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-16 10:10:00 | 409.00 | 2026-02-16 10:25:00 | 407.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-17 10:35:00 | 412.40 | 2026-02-17 10:40:00 | 411.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 10:55:00 | 406.20 | 2026-02-19 11:50:00 | 404.97 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-19 10:55:00 | 406.20 | 2026-02-19 13:05:00 | 406.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 10:45:00 | 400.95 | 2026-02-23 11:05:00 | 402.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 10:45:00 | 347.45 | 2026-03-13 10:50:00 | 349.12 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-21 09:40:00 | 352.10 | 2026-04-21 09:45:00 | 354.24 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-21 09:40:00 | 352.10 | 2026-04-21 09:50:00 | 352.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:50:00 | 353.20 | 2026-04-23 09:55:00 | 351.38 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-27 09:30:00 | 335.75 | 2026-04-27 09:50:00 | 338.02 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-27 09:30:00 | 335.75 | 2026-04-27 13:00:00 | 335.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:15:00 | 342.20 | 2026-04-28 10:25:00 | 340.93 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-05 09:35:00 | 350.90 | 2026-05-05 09:50:00 | 348.69 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2026-05-08 09:40:00 | 352.00 | 2026-05-08 09:50:00 | 353.22 | STOP_HIT | 1.00 | -0.35% |
