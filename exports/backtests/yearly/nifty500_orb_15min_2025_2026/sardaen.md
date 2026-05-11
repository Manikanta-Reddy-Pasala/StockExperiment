# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 591.90
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
| ENTRY1 | 75 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 15 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 60
- **Target hits / Stop hits / Partials:** 15 / 60 / 30
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 17.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 18 | 36.0% | 7 | 32 | 11 | 0.07% | 3.3% |
| BUY @ 2nd Alert (retest1) | 50 | 18 | 36.0% | 7 | 32 | 11 | 0.07% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 55 | 27 | 49.1% | 8 | 28 | 19 | 0.26% | 14.3% |
| SELL @ 2nd Alert (retest1) | 55 | 27 | 49.1% | 8 | 28 | 19 | 0.26% | 14.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 105 | 45 | 42.9% | 15 | 60 | 30 | 0.17% | 17.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-14 10:40:00 | 446.95 | 451.41 | 0.00 | ORB-short ORB[447.55,452.80] vol=2.6x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-05-14 10:55:00 | 448.97 | 449.66 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 11:05:00 | 452.45 | 455.06 | 0.00 | ORB-short ORB[453.00,459.00] vol=1.7x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-05-16 11:15:00 | 454.08 | 455.01 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:30:00 | 466.05 | 463.05 | 0.00 | ORB-long ORB[460.05,465.95] vol=2.9x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 09:35:00 | 469.65 | 463.52 | 0.00 | T1 1.5R @ 469.65 |
| Target hit | 2025-05-21 11:20:00 | 471.40 | 472.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2025-05-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:20:00 | 438.10 | 443.42 | 0.00 | ORB-short ORB[440.30,446.20] vol=1.7x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-05-27 10:45:00 | 440.47 | 442.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:50:00 | 440.70 | 437.68 | 0.00 | ORB-long ORB[433.85,439.20] vol=3.8x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-05-29 10:00:00 | 438.61 | 437.92 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:55:00 | 435.05 | 435.98 | 0.00 | ORB-short ORB[435.40,441.10] vol=2.2x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-05-30 11:20:00 | 436.11 | 435.99 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 11:05:00 | 440.50 | 438.32 | 0.00 | ORB-long ORB[434.50,440.30] vol=2.4x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-06-03 11:10:00 | 439.38 | 438.82 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:55:00 | 442.95 | 441.33 | 0.00 | ORB-long ORB[439.05,442.20] vol=2.6x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 10:05:00 | 444.93 | 441.96 | 0.00 | T1 1.5R @ 444.93 |
| Target hit | 2025-06-05 12:15:00 | 448.00 | 448.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 441.00 | 442.26 | 0.00 | ORB-short ORB[441.10,447.40] vol=2.4x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 11:30:00 | 439.36 | 441.74 | 0.00 | T1 1.5R @ 439.36 |
| Target hit | 2025-06-06 15:20:00 | 435.50 | 439.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:30:00 | 451.15 | 453.85 | 0.00 | ORB-short ORB[452.30,456.60] vol=1.8x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 09:35:00 | 448.36 | 453.45 | 0.00 | T1 1.5R @ 448.36 |
| Stop hit — per-position SL triggered | 2025-06-10 10:05:00 | 451.15 | 452.06 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:30:00 | 456.90 | 456.10 | 0.00 | ORB-long ORB[451.20,456.00] vol=4.7x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-06-11 09:35:00 | 454.82 | 456.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 11:05:00 | 450.85 | 447.96 | 0.00 | ORB-long ORB[440.00,446.20] vol=2.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-06-13 11:10:00 | 449.22 | 448.00 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 10:25:00 | 456.50 | 453.12 | 0.00 | ORB-long ORB[447.15,453.30] vol=3.2x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-06-17 10:40:00 | 455.05 | 453.24 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 454.85 | 452.47 | 0.00 | ORB-long ORB[448.05,453.20] vol=3.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-06-19 09:35:00 | 453.24 | 452.59 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 11:00:00 | 440.00 | 437.44 | 0.00 | ORB-long ORB[434.00,438.40] vol=2.5x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:05:00 | 441.73 | 438.49 | 0.00 | T1 1.5R @ 441.73 |
| Target hit | 2025-06-24 13:50:00 | 441.25 | 441.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-06-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:00:00 | 445.35 | 447.25 | 0.00 | ORB-short ORB[447.15,453.60] vol=3.3x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 13:05:00 | 443.52 | 446.86 | 0.00 | T1 1.5R @ 443.52 |
| Target hit | 2025-06-30 15:20:00 | 444.20 | 445.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 435.65 | 437.79 | 0.00 | ORB-short ORB[436.05,440.40] vol=2.2x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:50:00 | 433.33 | 437.41 | 0.00 | T1 1.5R @ 433.33 |
| Stop hit — per-position SL triggered | 2025-07-02 13:10:00 | 435.65 | 435.71 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:40:00 | 432.50 | 433.15 | 0.00 | ORB-short ORB[433.00,437.95] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-07-04 10:55:00 | 433.50 | 433.11 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 10:35:00 | 424.50 | 426.00 | 0.00 | ORB-short ORB[424.95,429.50] vol=1.5x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-07-09 10:45:00 | 425.69 | 425.97 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:05:00 | 447.45 | 446.09 | 0.00 | ORB-long ORB[444.05,447.30] vol=2.2x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 446.39 | 446.16 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:05:00 | 441.35 | 443.65 | 0.00 | ORB-short ORB[443.00,446.55] vol=3.2x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 439.66 | 443.04 | 0.00 | T1 1.5R @ 439.66 |
| Stop hit — per-position SL triggered | 2025-07-18 11:20:00 | 441.35 | 441.87 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:55:00 | 446.50 | 449.45 | 0.00 | ORB-short ORB[448.60,453.85] vol=2.2x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-07-22 11:25:00 | 447.66 | 448.93 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 11:00:00 | 438.90 | 443.84 | 0.00 | ORB-short ORB[444.35,449.40] vol=4.5x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-07-25 11:30:00 | 440.26 | 443.18 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:05:00 | 572.40 | 565.66 | 0.00 | ORB-long ORB[559.10,566.75] vol=1.7x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:35:00 | 576.57 | 568.68 | 0.00 | T1 1.5R @ 576.57 |
| Target hit | 2025-09-05 14:45:00 | 573.05 | 573.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 591.35 | 586.77 | 0.00 | ORB-long ORB[580.50,588.00] vol=1.6x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-09-10 09:50:00 | 588.00 | 587.30 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 09:45:00 | 581.00 | 583.45 | 0.00 | ORB-short ORB[582.10,587.80] vol=2.4x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-09-11 11:05:00 | 583.07 | 582.35 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:20:00 | 594.00 | 587.97 | 0.00 | ORB-long ORB[581.15,587.95] vol=5.1x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-09-25 10:30:00 | 591.64 | 588.60 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:50:00 | 560.10 | 555.56 | 0.00 | ORB-long ORB[550.00,557.00] vol=1.6x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:35:00 | 564.07 | 558.04 | 0.00 | T1 1.5R @ 564.07 |
| Target hit | 2025-10-03 15:20:00 | 577.70 | 570.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-10-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:30:00 | 580.65 | 576.76 | 0.00 | ORB-long ORB[572.85,579.35] vol=2.5x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-10-06 10:40:00 | 578.15 | 577.43 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 10:50:00 | 566.55 | 567.64 | 0.00 | ORB-short ORB[566.60,574.10] vol=1.6x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 11:50:00 | 564.01 | 566.84 | 0.00 | T1 1.5R @ 564.01 |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 566.55 | 566.46 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 557.60 | 560.06 | 0.00 | ORB-short ORB[558.90,564.10] vol=2.6x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 559.60 | 559.45 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:35:00 | 557.50 | 559.24 | 0.00 | ORB-short ORB[558.00,562.00] vol=1.7x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:45:00 | 554.94 | 558.36 | 0.00 | T1 1.5R @ 554.94 |
| Target hit | 2025-10-14 15:20:00 | 542.00 | 545.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-10-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:55:00 | 546.25 | 551.31 | 0.00 | ORB-short ORB[550.75,554.00] vol=2.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-10-16 11:30:00 | 547.99 | 550.66 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 546.30 | 543.62 | 0.00 | ORB-long ORB[540.60,545.00] vol=2.2x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-10-17 09:50:00 | 544.56 | 544.97 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 10:05:00 | 529.20 | 532.13 | 0.00 | ORB-short ORB[530.00,537.10] vol=1.6x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-11-03 10:20:00 | 531.17 | 532.16 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:50:00 | 540.70 | 544.71 | 0.00 | ORB-short ORB[544.40,549.50] vol=1.8x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:15:00 | 538.03 | 543.70 | 0.00 | T1 1.5R @ 538.03 |
| Stop hit — per-position SL triggered | 2025-11-04 10:50:00 | 540.70 | 543.03 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 10:55:00 | 522.60 | 526.96 | 0.00 | ORB-short ORB[525.80,529.50] vol=2.3x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:30:00 | 520.04 | 525.66 | 0.00 | T1 1.5R @ 520.04 |
| Target hit | 2025-11-13 15:20:00 | 514.80 | 520.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-11-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:25:00 | 520.70 | 516.22 | 0.00 | ORB-long ORB[512.15,516.00] vol=3.8x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-11-17 11:35:00 | 518.76 | 517.45 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:50:00 | 514.10 | 516.33 | 0.00 | ORB-short ORB[516.10,520.90] vol=2.3x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-11-18 10:55:00 | 515.64 | 515.05 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:45:00 | 501.50 | 504.53 | 0.00 | ORB-short ORB[502.30,508.15] vol=1.7x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:10:00 | 498.59 | 502.79 | 0.00 | T1 1.5R @ 498.59 |
| Stop hit — per-position SL triggered | 2025-11-21 11:50:00 | 501.50 | 501.35 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:50:00 | 476.55 | 482.96 | 0.00 | ORB-short ORB[483.10,488.60] vol=1.6x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:30:00 | 473.75 | 480.29 | 0.00 | T1 1.5R @ 473.75 |
| Stop hit — per-position SL triggered | 2025-11-24 13:00:00 | 476.55 | 477.46 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:35:00 | 491.00 | 488.13 | 0.00 | ORB-long ORB[484.00,490.50] vol=2.8x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 09:40:00 | 493.95 | 489.12 | 0.00 | T1 1.5R @ 493.95 |
| Stop hit — per-position SL triggered | 2025-11-28 09:45:00 | 491.00 | 489.37 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:10:00 | 488.75 | 494.15 | 0.00 | ORB-short ORB[495.25,500.45] vol=2.4x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-12-02 12:25:00 | 490.28 | 492.78 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 488.35 | 491.39 | 0.00 | ORB-short ORB[490.25,494.90] vol=1.6x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:05:00 | 485.85 | 489.01 | 0.00 | T1 1.5R @ 485.85 |
| Target hit | 2025-12-03 15:20:00 | 482.00 | 484.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:50:00 | 488.85 | 486.39 | 0.00 | ORB-long ORB[479.90,485.00] vol=2.7x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:05:00 | 492.06 | 489.64 | 0.00 | T1 1.5R @ 492.06 |
| Target hit | 2025-12-10 10:25:00 | 489.50 | 489.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2025-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:35:00 | 493.35 | 490.82 | 0.00 | ORB-long ORB[487.00,491.90] vol=2.5x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-12-12 09:45:00 | 491.55 | 496.11 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 515.50 | 515.89 | 0.00 | ORB-short ORB[516.50,521.70] vol=2.6x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:35:00 | 512.90 | 515.78 | 0.00 | T1 1.5R @ 512.90 |
| Target hit | 2025-12-16 15:20:00 | 505.50 | 512.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-12-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:05:00 | 499.75 | 503.16 | 0.00 | ORB-short ORB[504.00,509.00] vol=1.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-12-17 11:25:00 | 501.24 | 502.93 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:25:00 | 504.50 | 500.80 | 0.00 | ORB-long ORB[500.15,504.20] vol=1.8x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-12-18 11:05:00 | 502.78 | 501.15 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:45:00 | 517.00 | 514.41 | 0.00 | ORB-long ORB[511.30,514.70] vol=2.2x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:50:00 | 519.63 | 515.86 | 0.00 | T1 1.5R @ 519.63 |
| Target hit | 2025-12-23 10:40:00 | 522.60 | 524.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — SELL (started 2025-12-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:00:00 | 529.55 | 530.94 | 0.00 | ORB-short ORB[530.80,538.00] vol=1.6x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 10:20:00 | 527.02 | 529.87 | 0.00 | T1 1.5R @ 527.02 |
| Stop hit — per-position SL triggered | 2025-12-26 10:25:00 | 529.55 | 529.85 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 531.35 | 527.13 | 0.00 | ORB-long ORB[523.15,529.45] vol=2.2x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-12-29 10:05:00 | 529.39 | 529.25 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:15:00 | 520.65 | 517.55 | 0.00 | ORB-long ORB[511.40,517.25] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-12-30 10:35:00 | 518.54 | 517.86 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:45:00 | 523.70 | 516.54 | 0.00 | ORB-long ORB[511.30,516.50] vol=2.1x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-01-02 11:10:00 | 521.88 | 518.57 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:00:00 | 508.95 | 513.86 | 0.00 | ORB-short ORB[512.90,517.90] vol=2.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:10:00 | 506.64 | 511.92 | 0.00 | T1 1.5R @ 506.64 |
| Target hit | 2026-01-06 15:20:00 | 502.95 | 506.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2026-01-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:35:00 | 483.90 | 479.80 | 0.00 | ORB-long ORB[475.50,482.40] vol=2.0x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-01-28 12:20:00 | 482.36 | 481.70 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-02-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 09:35:00 | 496.00 | 492.71 | 0.00 | ORB-long ORB[489.15,495.40] vol=2.5x ATR=2.91 |
| Stop hit — per-position SL triggered | 2026-02-03 09:45:00 | 493.09 | 493.04 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-02-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:05:00 | 521.55 | 517.66 | 0.00 | ORB-long ORB[512.30,519.70] vol=2.1x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-02-10 11:35:00 | 519.53 | 517.83 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 497.40 | 493.99 | 0.00 | ORB-long ORB[491.40,497.25] vol=2.1x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 12:25:00 | 500.46 | 494.94 | 0.00 | T1 1.5R @ 500.46 |
| Stop hit — per-position SL triggered | 2026-02-13 13:30:00 | 497.40 | 495.63 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 498.30 | 495.15 | 0.00 | ORB-long ORB[492.65,497.50] vol=3.0x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 496.83 | 495.43 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 508.50 | 506.48 | 0.00 | ORB-long ORB[502.00,508.00] vol=2.2x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-17 10:10:00 | 506.78 | 507.15 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 514.55 | 510.42 | 0.00 | ORB-long ORB[504.05,508.15] vol=2.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-20 12:10:00 | 513.10 | 510.96 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 514.80 | 511.72 | 0.00 | ORB-long ORB[509.00,514.30] vol=3.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:40:00 | 517.39 | 513.06 | 0.00 | T1 1.5R @ 517.39 |
| Stop hit — per-position SL triggered | 2026-02-24 11:00:00 | 514.80 | 513.31 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 533.95 | 527.79 | 0.00 | ORB-long ORB[522.00,528.45] vol=3.8x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:55:00 | 538.07 | 530.52 | 0.00 | T1 1.5R @ 538.07 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 533.95 | 531.75 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-03-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:45:00 | 556.00 | 551.39 | 0.00 | ORB-long ORB[548.00,554.95] vol=3.5x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-03-10 10:50:00 | 553.77 | 551.52 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 556.55 | 552.69 | 0.00 | ORB-long ORB[544.85,552.80] vol=3.2x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-03-11 09:50:00 | 554.45 | 553.22 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-03-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:05:00 | 535.20 | 542.39 | 0.00 | ORB-short ORB[541.80,548.00] vol=1.5x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 531.84 | 541.10 | 0.00 | T1 1.5R @ 531.84 |
| Stop hit — per-position SL triggered | 2026-03-13 10:25:00 | 535.20 | 540.09 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:50:00 | 508.60 | 511.10 | 0.00 | ORB-short ORB[509.15,516.50] vol=1.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-03-27 11:30:00 | 510.82 | 510.65 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-04-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-02 09:35:00 | 510.45 | 512.94 | 0.00 | ORB-short ORB[511.40,517.60] vol=1.7x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 10:00:00 | 506.25 | 511.28 | 0.00 | T1 1.5R @ 506.25 |
| Target hit | 2026-04-02 11:15:00 | 509.25 | 508.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — BUY (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 593.95 | 590.32 | 0.00 | ORB-long ORB[585.95,592.00] vol=4.5x ATR=2.12 |
| Stop hit — per-position SL triggered | 2026-04-16 09:40:00 | 591.83 | 590.83 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-04-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:10:00 | 595.85 | 599.58 | 0.00 | ORB-short ORB[597.60,604.00] vol=2.2x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:05:00 | 593.17 | 599.05 | 0.00 | T1 1.5R @ 593.17 |
| Stop hit — per-position SL triggered | 2026-04-17 13:10:00 | 595.85 | 598.66 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 597.60 | 594.44 | 0.00 | ORB-long ORB[589.15,596.75] vol=2.5x ATR=2.15 |
| Stop hit — per-position SL triggered | 2026-04-23 10:30:00 | 595.45 | 594.82 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 623.95 | 619.27 | 0.00 | ORB-long ORB[614.40,621.25] vol=2.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 621.71 | 619.34 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 598.75 | 602.55 | 0.00 | ORB-short ORB[600.70,608.50] vol=1.7x ATR=2.90 |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 601.65 | 601.36 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 590.30 | 591.64 | 0.00 | ORB-short ORB[590.80,597.60] vol=1.9x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:20:00 | 587.24 | 591.11 | 0.00 | T1 1.5R @ 587.24 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 590.30 | 589.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-14 10:40:00 | 446.95 | 2025-05-14 10:55:00 | 448.97 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-05-16 11:05:00 | 452.45 | 2025-05-16 11:15:00 | 454.08 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-21 09:30:00 | 466.05 | 2025-05-21 09:35:00 | 469.65 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2025-05-21 09:30:00 | 466.05 | 2025-05-21 11:20:00 | 471.40 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2025-05-27 10:20:00 | 438.10 | 2025-05-27 10:45:00 | 440.47 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-05-29 09:50:00 | 440.70 | 2025-05-29 10:00:00 | 438.61 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-05-30 10:55:00 | 435.05 | 2025-05-30 11:20:00 | 436.11 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-03 11:05:00 | 440.50 | 2025-06-03 11:10:00 | 439.38 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-05 09:55:00 | 442.95 | 2025-06-05 10:05:00 | 444.93 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-06-05 09:55:00 | 442.95 | 2025-06-05 12:15:00 | 448.00 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-06-06 11:15:00 | 441.00 | 2025-06-06 11:30:00 | 439.36 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-06-06 11:15:00 | 441.00 | 2025-06-06 15:20:00 | 435.50 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2025-06-10 09:30:00 | 451.15 | 2025-06-10 09:35:00 | 448.36 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-06-10 09:30:00 | 451.15 | 2025-06-10 10:05:00 | 451.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-11 09:30:00 | 456.90 | 2025-06-11 09:35:00 | 454.82 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-06-13 11:05:00 | 450.85 | 2025-06-13 11:10:00 | 449.22 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-06-17 10:25:00 | 456.50 | 2025-06-17 10:40:00 | 455.05 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-19 09:30:00 | 454.85 | 2025-06-19 09:35:00 | 453.24 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-24 11:00:00 | 440.00 | 2025-06-24 11:05:00 | 441.73 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-24 11:00:00 | 440.00 | 2025-06-24 13:50:00 | 441.25 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-30 11:00:00 | 445.35 | 2025-06-30 13:05:00 | 443.52 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-30 11:00:00 | 445.35 | 2025-06-30 15:20:00 | 444.20 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-02 09:40:00 | 435.65 | 2025-07-02 09:50:00 | 433.33 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-07-02 09:40:00 | 435.65 | 2025-07-02 13:10:00 | 435.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-04 10:40:00 | 432.50 | 2025-07-04 10:55:00 | 433.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-09 10:35:00 | 424.50 | 2025-07-09 10:45:00 | 425.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-17 10:05:00 | 447.45 | 2025-07-17 10:15:00 | 446.39 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-18 10:05:00 | 441.35 | 2025-07-18 10:15:00 | 439.66 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-18 10:05:00 | 441.35 | 2025-07-18 11:20:00 | 441.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 10:55:00 | 446.50 | 2025-07-22 11:25:00 | 447.66 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-25 11:00:00 | 438.90 | 2025-07-25 11:30:00 | 440.26 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-05 10:05:00 | 572.40 | 2025-09-05 10:35:00 | 576.57 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-09-05 10:05:00 | 572.40 | 2025-09-05 14:45:00 | 573.05 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2025-09-10 09:40:00 | 591.35 | 2025-09-10 09:50:00 | 588.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2025-09-11 09:45:00 | 581.00 | 2025-09-11 11:05:00 | 583.07 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-25 10:20:00 | 594.00 | 2025-09-25 10:30:00 | 591.64 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-10-03 09:50:00 | 560.10 | 2025-10-03 10:35:00 | 564.07 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-10-03 09:50:00 | 560.10 | 2025-10-03 15:20:00 | 577.70 | TARGET_HIT | 0.50 | 3.14% |
| BUY | retest1 | 2025-10-06 10:30:00 | 580.65 | 2025-10-06 10:40:00 | 578.15 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-10-10 10:50:00 | 566.55 | 2025-10-10 11:50:00 | 564.01 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-10-10 10:50:00 | 566.55 | 2025-10-10 13:15:00 | 566.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-13 09:30:00 | 557.60 | 2025-10-13 09:35:00 | 559.60 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-14 09:35:00 | 557.50 | 2025-10-14 09:45:00 | 554.94 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-14 09:35:00 | 557.50 | 2025-10-14 15:20:00 | 542.00 | TARGET_HIT | 0.50 | 2.78% |
| SELL | retest1 | 2025-10-16 10:55:00 | 546.25 | 2025-10-16 11:30:00 | 547.99 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-17 09:30:00 | 546.30 | 2025-10-17 09:50:00 | 544.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-03 10:05:00 | 529.20 | 2025-11-03 10:20:00 | 531.17 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-04 09:50:00 | 540.70 | 2025-11-04 10:15:00 | 538.03 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-04 09:50:00 | 540.70 | 2025-11-04 10:50:00 | 540.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-13 10:55:00 | 522.60 | 2025-11-13 11:30:00 | 520.04 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-13 10:55:00 | 522.60 | 2025-11-13 15:20:00 | 514.80 | TARGET_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2025-11-17 10:25:00 | 520.70 | 2025-11-17 11:35:00 | 518.76 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-18 09:50:00 | 514.10 | 2025-11-18 10:55:00 | 515.64 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-21 09:45:00 | 501.50 | 2025-11-21 10:10:00 | 498.59 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-11-21 09:45:00 | 501.50 | 2025-11-21 11:50:00 | 501.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-24 10:50:00 | 476.55 | 2025-11-24 11:30:00 | 473.75 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-11-24 10:50:00 | 476.55 | 2025-11-24 13:00:00 | 476.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 09:35:00 | 491.00 | 2025-11-28 09:40:00 | 493.95 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-11-28 09:35:00 | 491.00 | 2025-11-28 09:45:00 | 491.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-02 11:10:00 | 488.75 | 2025-12-02 12:25:00 | 490.28 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-03 09:30:00 | 488.35 | 2025-12-03 10:05:00 | 485.85 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-03 09:30:00 | 488.35 | 2025-12-03 15:20:00 | 482.00 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2025-12-10 09:50:00 | 488.85 | 2025-12-10 10:05:00 | 492.06 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-12-10 09:50:00 | 488.85 | 2025-12-10 10:25:00 | 489.50 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2025-12-12 09:35:00 | 493.35 | 2025-12-12 09:45:00 | 491.55 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-12-16 11:00:00 | 515.50 | 2025-12-16 11:35:00 | 512.90 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-12-16 11:00:00 | 515.50 | 2025-12-16 15:20:00 | 505.50 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2025-12-17 11:05:00 | 499.75 | 2025-12-17 11:25:00 | 501.24 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-18 10:25:00 | 504.50 | 2025-12-18 11:05:00 | 502.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-23 09:45:00 | 517.00 | 2025-12-23 09:50:00 | 519.63 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-12-23 09:45:00 | 517.00 | 2025-12-23 10:40:00 | 522.60 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2025-12-26 10:00:00 | 529.55 | 2025-12-26 10:20:00 | 527.02 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-12-26 10:00:00 | 529.55 | 2025-12-26 10:25:00 | 529.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-29 09:30:00 | 531.35 | 2025-12-29 10:05:00 | 529.39 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-30 10:15:00 | 520.65 | 2025-12-30 10:35:00 | 518.54 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-01-02 10:45:00 | 523.70 | 2026-01-02 11:10:00 | 521.88 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-06 11:00:00 | 508.95 | 2026-01-06 11:10:00 | 506.64 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-06 11:00:00 | 508.95 | 2026-01-06 15:20:00 | 502.95 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2026-01-28 10:35:00 | 483.90 | 2026-01-28 12:20:00 | 482.36 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-03 09:35:00 | 496.00 | 2026-02-03 09:45:00 | 493.09 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2026-02-10 11:05:00 | 521.55 | 2026-02-10 11:35:00 | 519.53 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-13 11:00:00 | 497.40 | 2026-02-13 12:25:00 | 500.46 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-13 11:00:00 | 497.40 | 2026-02-13 13:30:00 | 497.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 11:05:00 | 498.30 | 2026-02-16 11:25:00 | 496.83 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-17 09:45:00 | 508.50 | 2026-02-17 10:10:00 | 506.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-20 10:55:00 | 514.55 | 2026-02-20 12:10:00 | 513.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-24 10:25:00 | 514.80 | 2026-02-24 10:40:00 | 517.39 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-24 10:25:00 | 514.80 | 2026-02-24 11:00:00 | 514.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:45:00 | 533.95 | 2026-03-05 10:55:00 | 538.07 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-03-05 10:45:00 | 533.95 | 2026-03-05 11:45:00 | 533.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:45:00 | 556.00 | 2026-03-10 10:50:00 | 553.77 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-11 09:40:00 | 556.55 | 2026-03-11 09:50:00 | 554.45 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-13 10:05:00 | 535.20 | 2026-03-13 10:20:00 | 531.84 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-13 10:05:00 | 535.20 | 2026-03-13 10:25:00 | 535.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:50:00 | 508.60 | 2026-03-27 11:30:00 | 510.82 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-02 09:35:00 | 510.45 | 2026-04-02 10:00:00 | 506.25 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2026-04-02 09:35:00 | 510.45 | 2026-04-02 11:15:00 | 509.25 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2026-04-16 09:35:00 | 593.95 | 2026-04-16 09:40:00 | 591.83 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-17 11:10:00 | 595.85 | 2026-04-17 12:05:00 | 593.17 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-17 11:10:00 | 595.85 | 2026-04-17 13:10:00 | 595.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:15:00 | 597.60 | 2026-04-23 10:30:00 | 595.45 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-28 11:00:00 | 623.95 | 2026-04-28 11:05:00 | 621.71 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-29 09:40:00 | 598.75 | 2026-04-29 10:15:00 | 601.65 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-05-05 10:35:00 | 590.30 | 2026-05-05 11:20:00 | 587.24 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-05 10:35:00 | 590.30 | 2026-05-05 12:50:00 | 590.30 | STOP_HIT | 0.50 | 0.00% |
