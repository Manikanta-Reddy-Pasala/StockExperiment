# RHI MAGNESITA INDIA LTD. (RHIM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-02-04 15:25:00 (13873 bars)
- **Last close:** 446.00
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 13 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 61
- **Target hits / Stop hits / Partials:** 13 / 61 / 35
- **Avg / median % per leg:** 0.33% / 0.00%
- **Sum % (uncompounded):** 36.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 25 | 44.6% | 8 | 31 | 17 | 0.46% | 25.7% |
| BUY @ 2nd Alert (retest1) | 56 | 25 | 44.6% | 8 | 31 | 17 | 0.46% | 25.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 23 | 43.4% | 5 | 30 | 18 | 0.20% | 10.7% |
| SELL @ 2nd Alert (retest1) | 53 | 23 | 43.4% | 5 | 30 | 18 | 0.20% | 10.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 109 | 48 | 44.0% | 13 | 61 | 35 | 0.33% | 36.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:50:00 | 473.25 | 469.81 | 0.00 | ORB-long ORB[466.00,470.95] vol=2.1x ATR=2.26 |
| Stop hit — per-position SL triggered | 2025-05-19 09:55:00 | 470.99 | 470.01 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 10:40:00 | 466.20 | 464.05 | 0.00 | ORB-long ORB[459.40,465.80] vol=1.7x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 10:50:00 | 468.32 | 465.49 | 0.00 | T1 1.5R @ 468.32 |
| Target hit | 2025-05-22 15:20:00 | 471.55 | 468.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 11:00:00 | 464.90 | 469.01 | 0.00 | ORB-short ORB[468.20,473.70] vol=5.3x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-05-23 11:20:00 | 466.05 | 468.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:20:00 | 469.05 | 467.17 | 0.00 | ORB-long ORB[464.10,468.20] vol=2.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-05-26 10:25:00 | 467.41 | 467.27 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:45:00 | 472.35 | 468.82 | 0.00 | ORB-long ORB[465.50,470.10] vol=6.9x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:50:00 | 475.26 | 469.54 | 0.00 | T1 1.5R @ 475.26 |
| Stop hit — per-position SL triggered | 2025-05-28 11:00:00 | 472.35 | 469.78 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:45:00 | 454.45 | 457.38 | 0.00 | ORB-short ORB[457.75,464.00] vol=1.6x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-05-30 10:50:00 | 455.87 | 457.20 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 11:00:00 | 457.80 | 458.21 | 0.00 | ORB-short ORB[458.00,463.35] vol=3.1x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:05:00 | 456.38 | 458.11 | 0.00 | T1 1.5R @ 456.38 |
| Target hit | 2025-06-02 13:05:00 | 457.35 | 457.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 436.60 | 438.68 | 0.00 | ORB-short ORB[437.05,443.00] vol=2.2x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-06-04 09:45:00 | 437.98 | 438.01 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:20:00 | 447.00 | 443.65 | 0.00 | ORB-long ORB[438.75,444.55] vol=1.7x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 10:25:00 | 449.14 | 448.20 | 0.00 | T1 1.5R @ 449.14 |
| Target hit | 2025-06-05 10:45:00 | 455.75 | 456.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 449.00 | 452.49 | 0.00 | ORB-short ORB[449.70,454.75] vol=2.1x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-06-06 13:05:00 | 450.33 | 451.71 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:30:00 | 476.00 | 468.27 | 0.00 | ORB-long ORB[459.75,466.20] vol=9.9x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-06-12 10:35:00 | 472.84 | 469.49 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:05:00 | 495.15 | 491.39 | 0.00 | ORB-long ORB[485.10,492.45] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-06-20 11:50:00 | 493.01 | 491.56 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:05:00 | 490.80 | 486.50 | 0.00 | ORB-long ORB[483.45,488.00] vol=3.2x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 11:35:00 | 493.02 | 488.30 | 0.00 | T1 1.5R @ 493.02 |
| Stop hit — per-position SL triggered | 2025-06-23 12:20:00 | 490.80 | 489.02 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:05:00 | 479.70 | 482.90 | 0.00 | ORB-short ORB[481.80,487.05] vol=1.5x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:20:00 | 477.33 | 481.69 | 0.00 | T1 1.5R @ 477.33 |
| Stop hit — per-position SL triggered | 2025-06-26 14:45:00 | 479.70 | 479.29 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:00:00 | 469.60 | 472.47 | 0.00 | ORB-short ORB[472.55,476.30] vol=1.6x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:40:00 | 467.21 | 471.32 | 0.00 | T1 1.5R @ 467.21 |
| Target hit | 2025-07-01 15:20:00 | 463.90 | 467.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 10:05:00 | 471.60 | 467.55 | 0.00 | ORB-long ORB[462.60,466.85] vol=4.2x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:10:00 | 474.68 | 470.91 | 0.00 | T1 1.5R @ 474.68 |
| Stop hit — per-position SL triggered | 2025-07-02 10:30:00 | 471.60 | 471.54 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:05:00 | 475.80 | 472.10 | 0.00 | ORB-long ORB[465.00,472.00] vol=1.5x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-07-03 10:30:00 | 473.64 | 472.72 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:15:00 | 467.25 | 469.72 | 0.00 | ORB-short ORB[468.90,473.65] vol=1.8x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:30:00 | 465.25 | 469.56 | 0.00 | T1 1.5R @ 465.25 |
| Target hit | 2025-07-08 15:20:00 | 466.20 | 467.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:15:00 | 472.90 | 470.68 | 0.00 | ORB-long ORB[467.00,471.40] vol=2.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-07-09 10:40:00 | 471.48 | 470.90 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:30:00 | 484.00 | 481.67 | 0.00 | ORB-long ORB[478.00,483.50] vol=1.5x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-10 11:00:00 | 482.21 | 482.55 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:50:00 | 484.60 | 480.98 | 0.00 | ORB-long ORB[478.05,483.20] vol=2.1x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-07-11 10:00:00 | 482.59 | 481.38 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:15:00 | 485.80 | 483.56 | 0.00 | ORB-long ORB[479.05,485.00] vol=2.8x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-07-16 10:40:00 | 484.22 | 484.08 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:20:00 | 481.00 | 482.54 | 0.00 | ORB-short ORB[481.60,486.90] vol=1.5x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:35:00 | 479.11 | 482.19 | 0.00 | T1 1.5R @ 479.11 |
| Stop hit — per-position SL triggered | 2025-07-17 12:40:00 | 481.00 | 481.12 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 475.15 | 478.65 | 0.00 | ORB-short ORB[478.55,481.40] vol=3.5x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 473.20 | 477.83 | 0.00 | T1 1.5R @ 473.20 |
| Stop hit — per-position SL triggered | 2025-07-18 13:20:00 | 475.15 | 475.77 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:55:00 | 484.45 | 482.91 | 0.00 | ORB-long ORB[478.70,483.60] vol=3.3x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-07-21 10:00:00 | 482.25 | 482.90 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 518.70 | 522.43 | 0.00 | ORB-short ORB[520.00,527.50] vol=2.5x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:35:00 | 515.73 | 521.81 | 0.00 | T1 1.5R @ 515.73 |
| Target hit | 2025-08-06 15:20:00 | 500.50 | 510.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 500.00 | 497.13 | 0.00 | ORB-long ORB[491.20,495.55] vol=7.2x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-08-19 10:05:00 | 498.14 | 497.52 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:40:00 | 505.05 | 502.71 | 0.00 | ORB-long ORB[497.35,503.25] vol=4.0x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 10:10:00 | 508.72 | 504.67 | 0.00 | T1 1.5R @ 508.72 |
| Stop hit — per-position SL triggered | 2025-08-21 10:20:00 | 505.05 | 504.75 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:45:00 | 489.80 | 494.52 | 0.00 | ORB-short ORB[495.95,502.45] vol=1.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-08-22 11:00:00 | 491.14 | 492.36 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 10:55:00 | 472.00 | 474.62 | 0.00 | ORB-short ORB[472.70,477.80] vol=3.2x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 13:30:00 | 470.24 | 473.14 | 0.00 | T1 1.5R @ 470.24 |
| Stop hit — per-position SL triggered | 2025-09-08 14:40:00 | 472.00 | 472.49 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 474.90 | 472.30 | 0.00 | ORB-long ORB[469.95,473.95] vol=2.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-09-10 09:50:00 | 473.57 | 472.74 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:45:00 | 478.00 | 476.30 | 0.00 | ORB-long ORB[472.80,477.00] vol=3.9x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-09-11 09:55:00 | 476.48 | 476.44 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:15:00 | 475.70 | 478.77 | 0.00 | ORB-short ORB[475.90,481.20] vol=1.7x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-09-15 11:25:00 | 477.00 | 478.52 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:00:00 | 474.85 | 475.92 | 0.00 | ORB-short ORB[475.25,478.05] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-09-16 11:30:00 | 475.83 | 475.89 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:00:00 | 489.00 | 485.84 | 0.00 | ORB-long ORB[481.00,488.00] vol=1.7x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:05:00 | 491.58 | 488.04 | 0.00 | T1 1.5R @ 491.58 |
| Stop hit — per-position SL triggered | 2025-09-17 10:20:00 | 489.00 | 488.45 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:05:00 | 479.80 | 480.76 | 0.00 | ORB-short ORB[480.25,483.85] vol=3.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-09-22 11:50:00 | 480.89 | 480.56 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:05:00 | 466.95 | 469.07 | 0.00 | ORB-short ORB[468.25,473.90] vol=2.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-09-24 11:10:00 | 467.86 | 469.02 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:45:00 | 455.15 | 458.25 | 0.00 | ORB-short ORB[457.45,463.40] vol=1.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-09-25 11:15:00 | 456.44 | 458.03 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:35:00 | 449.05 | 446.77 | 0.00 | ORB-long ORB[443.00,447.35] vol=2.1x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-09-29 10:25:00 | 447.13 | 447.70 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 11:10:00 | 444.00 | 439.85 | 0.00 | ORB-long ORB[437.45,442.15] vol=2.9x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 12:45:00 | 446.93 | 441.27 | 0.00 | T1 1.5R @ 446.93 |
| Stop hit — per-position SL triggered | 2025-09-30 13:00:00 | 444.00 | 441.40 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:30:00 | 443.50 | 442.17 | 0.00 | ORB-long ORB[438.35,442.60] vol=4.3x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:35:00 | 446.56 | 443.56 | 0.00 | T1 1.5R @ 446.56 |
| Stop hit — per-position SL triggered | 2025-10-01 09:50:00 | 443.50 | 443.87 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:30:00 | 444.30 | 446.91 | 0.00 | ORB-short ORB[445.00,449.60] vol=1.8x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-10-03 10:35:00 | 445.64 | 446.54 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:50:00 | 446.00 | 442.67 | 0.00 | ORB-long ORB[440.00,443.00] vol=2.5x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:55:00 | 447.78 | 444.93 | 0.00 | T1 1.5R @ 447.78 |
| Target hit | 2025-10-07 11:45:00 | 490.75 | 490.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2025-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:40:00 | 456.25 | 453.99 | 0.00 | ORB-long ORB[451.10,455.10] vol=1.5x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:45:00 | 458.26 | 455.11 | 0.00 | T1 1.5R @ 458.26 |
| Stop hit — per-position SL triggered | 2025-10-17 09:55:00 | 456.25 | 455.45 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:50:00 | 451.00 | 448.38 | 0.00 | ORB-long ORB[445.80,450.60] vol=1.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-10-20 09:55:00 | 449.30 | 448.53 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:20:00 | 448.20 | 449.98 | 0.00 | ORB-short ORB[449.10,453.10] vol=3.0x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-10-28 11:10:00 | 449.32 | 449.86 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:55:00 | 451.50 | 449.83 | 0.00 | ORB-long ORB[447.40,451.30] vol=1.9x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:00:00 | 453.47 | 459.65 | 0.00 | T1 1.5R @ 453.47 |
| Target hit | 2025-10-29 10:10:00 | 460.60 | 460.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — SELL (started 2025-11-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:55:00 | 468.75 | 471.84 | 0.00 | ORB-short ORB[474.50,477.15] vol=2.0x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 466.74 | 471.45 | 0.00 | T1 1.5R @ 466.74 |
| Stop hit — per-position SL triggered | 2025-11-06 12:20:00 | 468.75 | 469.83 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:35:00 | 467.10 | 466.33 | 0.00 | ORB-long ORB[463.05,466.55] vol=2.6x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:20:00 | 469.50 | 466.91 | 0.00 | T1 1.5R @ 469.50 |
| Target hit | 2025-11-12 15:20:00 | 480.05 | 478.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-11-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:35:00 | 489.35 | 486.33 | 0.00 | ORB-long ORB[483.00,488.35] vol=3.5x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-11-17 10:00:00 | 487.26 | 486.96 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:30:00 | 478.60 | 479.53 | 0.00 | ORB-short ORB[479.70,483.50] vol=4.6x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-11-19 10:50:00 | 480.07 | 479.44 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 480.85 | 482.13 | 0.00 | ORB-short ORB[481.15,485.00] vol=1.5x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 09:35:00 | 478.59 | 481.58 | 0.00 | T1 1.5R @ 478.59 |
| Stop hit — per-position SL triggered | 2025-11-20 09:40:00 | 480.85 | 481.53 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:50:00 | 467.00 | 467.56 | 0.00 | ORB-short ORB[470.45,473.00] vol=3.0x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-11-21 11:55:00 | 468.33 | 467.41 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:45:00 | 467.15 | 464.75 | 0.00 | ORB-long ORB[460.70,466.50] vol=1.7x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:20:00 | 469.56 | 466.37 | 0.00 | T1 1.5R @ 469.56 |
| Target hit | 2025-11-26 15:20:00 | 473.95 | 469.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-12-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:05:00 | 461.30 | 462.45 | 0.00 | ORB-short ORB[462.00,465.15] vol=1.9x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 12:05:00 | 459.46 | 461.73 | 0.00 | T1 1.5R @ 459.46 |
| Stop hit — per-position SL triggered | 2025-12-02 13:30:00 | 461.30 | 461.72 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 09:45:00 | 466.35 | 463.97 | 0.00 | ORB-long ORB[462.00,465.90] vol=2.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-12-03 09:55:00 | 464.80 | 464.01 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 454.45 | 452.53 | 0.00 | ORB-long ORB[450.05,453.95] vol=1.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-12-05 09:35:00 | 452.86 | 452.54 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 445.70 | 449.10 | 0.00 | ORB-short ORB[447.00,452.25] vol=4.2x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 12:35:00 | 443.22 | 447.70 | 0.00 | T1 1.5R @ 443.22 |
| Target hit | 2025-12-10 15:20:00 | 439.40 | 444.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-12-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:20:00 | 440.85 | 441.79 | 0.00 | ORB-short ORB[441.60,445.55] vol=2.1x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:45:00 | 439.08 | 441.44 | 0.00 | T1 1.5R @ 439.08 |
| Stop hit — per-position SL triggered | 2025-12-16 12:40:00 | 440.85 | 440.99 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:40:00 | 442.85 | 440.66 | 0.00 | ORB-long ORB[436.55,441.90] vol=3.7x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-12-17 10:00:00 | 441.43 | 440.97 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:00:00 | 435.80 | 438.25 | 0.00 | ORB-short ORB[438.80,442.35] vol=2.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-12-18 10:05:00 | 436.99 | 438.25 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:00:00 | 457.00 | 455.82 | 0.00 | ORB-long ORB[450.00,454.80] vol=6.5x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-12-23 10:05:00 | 455.46 | 455.78 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:30:00 | 450.45 | 447.86 | 0.00 | ORB-long ORB[444.35,448.25] vol=1.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-12-26 09:45:00 | 449.12 | 448.59 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:15:00 | 446.20 | 450.01 | 0.00 | ORB-short ORB[446.50,451.45] vol=2.9x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-12-29 11:30:00 | 447.38 | 449.88 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 444.90 | 447.09 | 0.00 | ORB-short ORB[445.80,450.00] vol=2.8x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:35:00 | 443.34 | 446.28 | 0.00 | T1 1.5R @ 443.34 |
| Stop hit — per-position SL triggered | 2025-12-30 11:45:00 | 444.90 | 445.85 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 09:35:00 | 451.95 | 448.00 | 0.00 | ORB-long ORB[444.55,448.50] vol=1.8x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 09:50:00 | 454.17 | 450.09 | 0.00 | T1 1.5R @ 454.17 |
| Target hit | 2025-12-31 11:20:00 | 456.40 | 456.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 67 — SELL (started 2026-01-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:05:00 | 475.75 | 476.98 | 0.00 | ORB-short ORB[476.05,482.80] vol=2.8x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:20:00 | 472.96 | 476.60 | 0.00 | T1 1.5R @ 472.96 |
| Stop hit — per-position SL triggered | 2026-01-08 10:40:00 | 475.75 | 476.44 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:50:00 | 454.20 | 457.43 | 0.00 | ORB-short ORB[454.90,460.15] vol=2.1x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:00:00 | 452.33 | 456.93 | 0.00 | T1 1.5R @ 452.33 |
| Stop hit — per-position SL triggered | 2026-01-16 11:05:00 | 454.20 | 456.83 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:20:00 | 424.45 | 429.22 | 0.00 | ORB-short ORB[428.20,434.15] vol=1.8x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:35:00 | 419.96 | 428.08 | 0.00 | T1 1.5R @ 419.96 |
| Stop hit — per-position SL triggered | 2026-01-21 11:10:00 | 424.45 | 427.19 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:30:00 | 429.35 | 426.68 | 0.00 | ORB-long ORB[423.00,428.55] vol=1.6x ATR=2.18 |
| Stop hit — per-position SL triggered | 2026-01-22 10:00:00 | 427.17 | 427.23 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 09:35:00 | 428.40 | 432.92 | 0.00 | ORB-short ORB[431.10,435.60] vol=1.7x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-01-23 09:45:00 | 430.36 | 431.59 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:15:00 | 419.85 | 415.77 | 0.00 | ORB-long ORB[410.60,414.45] vol=2.1x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 10:20:00 | 422.31 | 417.09 | 0.00 | T1 1.5R @ 422.31 |
| Stop hit — per-position SL triggered | 2026-01-28 10:35:00 | 419.85 | 417.47 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:40:00 | 418.20 | 414.71 | 0.00 | ORB-long ORB[409.10,413.70] vol=3.3x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:50:00 | 421.06 | 421.34 | 0.00 | T1 1.5R @ 421.06 |
| Target hit | 2026-01-30 10:15:00 | 437.50 | 438.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — SELL (started 2026-02-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-03 09:30:00 | 436.95 | 440.56 | 0.00 | ORB-short ORB[438.45,443.60] vol=2.2x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 09:40:00 | 433.01 | 439.32 | 0.00 | T1 1.5R @ 433.01 |
| Stop hit — per-position SL triggered | 2026-02-03 11:10:00 | 436.95 | 435.38 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-19 09:50:00 | 473.25 | 2025-05-19 09:55:00 | 470.99 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-05-22 10:40:00 | 466.20 | 2025-05-22 10:50:00 | 468.32 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-22 10:40:00 | 466.20 | 2025-05-22 15:20:00 | 471.55 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2025-05-23 11:00:00 | 464.90 | 2025-05-23 11:20:00 | 466.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-26 10:20:00 | 469.05 | 2025-05-26 10:25:00 | 467.41 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-28 10:45:00 | 472.35 | 2025-05-28 10:50:00 | 475.26 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-05-28 10:45:00 | 472.35 | 2025-05-28 11:00:00 | 472.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 10:45:00 | 454.45 | 2025-05-30 10:50:00 | 455.87 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-06-02 11:00:00 | 457.80 | 2025-06-02 11:05:00 | 456.38 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-06-02 11:00:00 | 457.80 | 2025-06-02 13:05:00 | 457.35 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-06-04 09:30:00 | 436.60 | 2025-06-04 09:45:00 | 437.98 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-05 10:20:00 | 447.00 | 2025-06-05 10:25:00 | 449.14 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-06-05 10:20:00 | 447.00 | 2025-06-05 10:45:00 | 455.75 | TARGET_HIT | 0.50 | 1.96% |
| SELL | retest1 | 2025-06-06 11:15:00 | 449.00 | 2025-06-06 13:05:00 | 450.33 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-12 10:30:00 | 476.00 | 2025-06-12 10:35:00 | 472.84 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2025-06-20 11:05:00 | 495.15 | 2025-06-20 11:50:00 | 493.01 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-23 11:05:00 | 490.80 | 2025-06-23 11:35:00 | 493.02 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-06-23 11:05:00 | 490.80 | 2025-06-23 12:20:00 | 490.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 10:05:00 | 479.70 | 2025-06-26 10:20:00 | 477.33 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-06-26 10:05:00 | 479.70 | 2025-06-26 14:45:00 | 479.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:00:00 | 469.60 | 2025-07-01 10:40:00 | 467.21 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-07-01 10:00:00 | 469.60 | 2025-07-01 15:20:00 | 463.90 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2025-07-02 10:05:00 | 471.60 | 2025-07-02 10:10:00 | 474.68 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-07-02 10:05:00 | 471.60 | 2025-07-02 10:30:00 | 471.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 10:05:00 | 475.80 | 2025-07-03 10:30:00 | 473.64 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-07-08 11:15:00 | 467.25 | 2025-07-08 11:30:00 | 465.25 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-08 11:15:00 | 467.25 | 2025-07-08 15:20:00 | 466.20 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-07-09 10:15:00 | 472.90 | 2025-07-09 10:40:00 | 471.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-10 10:30:00 | 484.00 | 2025-07-10 11:00:00 | 482.21 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-07-11 09:50:00 | 484.60 | 2025-07-11 10:00:00 | 482.59 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-07-16 10:15:00 | 485.80 | 2025-07-16 10:40:00 | 484.22 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-17 10:20:00 | 481.00 | 2025-07-17 10:35:00 | 479.11 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-17 10:20:00 | 481.00 | 2025-07-17 12:40:00 | 481.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 09:45:00 | 475.15 | 2025-07-18 10:15:00 | 473.20 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-18 09:45:00 | 475.15 | 2025-07-18 13:20:00 | 475.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 09:55:00 | 484.45 | 2025-07-21 10:00:00 | 482.25 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-08-06 09:30:00 | 518.70 | 2025-08-06 09:35:00 | 515.73 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-08-06 09:30:00 | 518.70 | 2025-08-06 15:20:00 | 500.50 | TARGET_HIT | 0.50 | 3.51% |
| BUY | retest1 | 2025-08-19 09:50:00 | 500.00 | 2025-08-19 10:05:00 | 498.14 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-08-21 09:40:00 | 505.05 | 2025-08-21 10:10:00 | 508.72 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-08-21 09:40:00 | 505.05 | 2025-08-21 10:20:00 | 505.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 10:45:00 | 489.80 | 2025-08-22 11:00:00 | 491.14 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-08 10:55:00 | 472.00 | 2025-09-08 13:30:00 | 470.24 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-08 10:55:00 | 472.00 | 2025-09-08 14:40:00 | 472.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-10 09:35:00 | 474.90 | 2025-09-10 09:50:00 | 473.57 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-11 09:45:00 | 478.00 | 2025-09-11 09:55:00 | 476.48 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-15 11:15:00 | 475.70 | 2025-09-15 11:25:00 | 477.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-16 11:00:00 | 474.85 | 2025-09-16 11:30:00 | 475.83 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-17 10:00:00 | 489.00 | 2025-09-17 10:05:00 | 491.58 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-09-17 10:00:00 | 489.00 | 2025-09-17 10:20:00 | 489.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-22 11:05:00 | 479.80 | 2025-09-22 11:50:00 | 480.89 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-24 11:05:00 | 466.95 | 2025-09-24 11:10:00 | 467.86 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-25 10:45:00 | 455.15 | 2025-09-25 11:15:00 | 456.44 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-29 09:35:00 | 449.05 | 2025-09-29 10:25:00 | 447.13 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-09-30 11:10:00 | 444.00 | 2025-09-30 12:45:00 | 446.93 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-09-30 11:10:00 | 444.00 | 2025-09-30 13:00:00 | 444.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 09:30:00 | 443.50 | 2025-10-01 09:35:00 | 446.56 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-10-01 09:30:00 | 443.50 | 2025-10-01 09:50:00 | 443.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-03 10:30:00 | 444.30 | 2025-10-03 10:35:00 | 445.64 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-07 09:50:00 | 446.00 | 2025-10-07 09:55:00 | 447.78 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-07 09:50:00 | 446.00 | 2025-10-07 11:45:00 | 490.75 | TARGET_HIT | 0.50 | 10.03% |
| BUY | retest1 | 2025-10-17 09:40:00 | 456.25 | 2025-10-17 09:45:00 | 458.26 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-17 09:40:00 | 456.25 | 2025-10-17 09:55:00 | 456.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 09:50:00 | 451.00 | 2025-10-20 09:55:00 | 449.30 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-10-28 10:20:00 | 448.20 | 2025-10-28 11:10:00 | 449.32 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-29 09:55:00 | 451.50 | 2025-10-29 10:00:00 | 453.47 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-29 09:55:00 | 451.50 | 2025-10-29 10:10:00 | 460.60 | TARGET_HIT | 0.50 | 2.02% |
| SELL | retest1 | 2025-11-06 10:55:00 | 468.75 | 2025-11-06 11:15:00 | 466.74 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-06 10:55:00 | 468.75 | 2025-11-06 12:20:00 | 468.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 09:35:00 | 467.10 | 2025-11-12 10:20:00 | 469.50 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-11-12 09:35:00 | 467.10 | 2025-11-12 15:20:00 | 480.05 | TARGET_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2025-11-17 09:35:00 | 489.35 | 2025-11-17 10:00:00 | 487.26 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-11-19 10:30:00 | 478.60 | 2025-11-19 10:50:00 | 480.07 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-20 09:30:00 | 480.85 | 2025-11-20 09:35:00 | 478.59 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-11-20 09:30:00 | 480.85 | 2025-11-20 09:40:00 | 480.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 10:50:00 | 467.00 | 2025-11-21 11:55:00 | 468.33 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-26 10:45:00 | 467.15 | 2025-11-26 11:20:00 | 469.56 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-26 10:45:00 | 467.15 | 2025-11-26 15:20:00 | 473.95 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2025-12-02 11:05:00 | 461.30 | 2025-12-02 12:05:00 | 459.46 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-02 11:05:00 | 461.30 | 2025-12-02 13:30:00 | 461.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-03 09:45:00 | 466.35 | 2025-12-03 09:55:00 | 464.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-05 09:30:00 | 454.45 | 2025-12-05 09:35:00 | 452.86 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-10 10:45:00 | 445.70 | 2025-12-10 12:35:00 | 443.22 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-12-10 10:45:00 | 445.70 | 2025-12-10 15:20:00 | 439.40 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2025-12-16 10:20:00 | 440.85 | 2025-12-16 11:45:00 | 439.08 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-16 10:20:00 | 440.85 | 2025-12-16 12:40:00 | 440.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 09:40:00 | 442.85 | 2025-12-17 10:00:00 | 441.43 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-18 10:00:00 | 435.80 | 2025-12-18 10:05:00 | 436.99 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-23 10:00:00 | 457.00 | 2025-12-23 10:05:00 | 455.46 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-26 09:30:00 | 450.45 | 2025-12-26 09:45:00 | 449.12 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-29 11:15:00 | 446.20 | 2025-12-29 11:30:00 | 447.38 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-30 10:50:00 | 444.90 | 2025-12-30 11:35:00 | 443.34 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-30 10:50:00 | 444.90 | 2025-12-30 11:45:00 | 444.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 09:35:00 | 451.95 | 2025-12-31 09:50:00 | 454.17 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-31 09:35:00 | 451.95 | 2025-12-31 11:20:00 | 456.40 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2026-01-08 10:05:00 | 475.75 | 2026-01-08 10:20:00 | 472.96 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-01-08 10:05:00 | 475.75 | 2026-01-08 10:40:00 | 475.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-16 10:50:00 | 454.20 | 2026-01-16 11:00:00 | 452.33 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-16 10:50:00 | 454.20 | 2026-01-16 11:05:00 | 454.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-21 10:20:00 | 424.45 | 2026-01-21 10:35:00 | 419.96 | PARTIAL | 0.50 | 1.06% |
| SELL | retest1 | 2026-01-21 10:20:00 | 424.45 | 2026-01-21 11:10:00 | 424.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-22 09:30:00 | 429.35 | 2026-01-22 10:00:00 | 427.17 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-01-23 09:35:00 | 428.40 | 2026-01-23 09:45:00 | 430.36 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-01-28 10:15:00 | 419.85 | 2026-01-28 10:20:00 | 422.31 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-01-28 10:15:00 | 419.85 | 2026-01-28 10:35:00 | 419.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 09:40:00 | 418.20 | 2026-01-30 09:50:00 | 421.06 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-01-30 09:40:00 | 418.20 | 2026-01-30 10:15:00 | 437.50 | TARGET_HIT | 0.50 | 4.62% |
| SELL | retest1 | 2026-02-03 09:30:00 | 436.95 | 2026-02-03 09:40:00 | 433.01 | PARTIAL | 0.50 | 0.90% |
| SELL | retest1 | 2026-02-03 09:30:00 | 436.95 | 2026-02-03 11:10:00 | 436.95 | STOP_HIT | 0.50 | 0.00% |
