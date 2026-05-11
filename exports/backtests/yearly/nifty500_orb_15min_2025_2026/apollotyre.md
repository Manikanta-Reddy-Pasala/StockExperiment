# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-03-06 15:25:00 (15463 bars)
- **Last close:** 439.95
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
| TARGET_HIT | 8 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 67
- **Target hits / Stop hits / Partials:** 8 / 67 / 30
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 5.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 19 | 31.1% | 3 | 42 | 16 | 0.01% | 0.7% |
| BUY @ 2nd Alert (retest1) | 61 | 19 | 31.1% | 3 | 42 | 16 | 0.01% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 19 | 43.2% | 5 | 25 | 14 | 0.12% | 5.3% |
| SELL @ 2nd Alert (retest1) | 44 | 19 | 43.2% | 5 | 25 | 14 | 0.12% | 5.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 105 | 38 | 36.2% | 8 | 67 | 30 | 0.06% | 5.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 495.70 | 493.67 | 0.00 | ORB-long ORB[491.00,495.00] vol=2.1x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 09:45:00 | 497.59 | 494.65 | 0.00 | T1 1.5R @ 497.59 |
| Stop hit — per-position SL triggered | 2025-05-23 10:05:00 | 495.70 | 494.87 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:45:00 | 495.25 | 491.46 | 0.00 | ORB-long ORB[488.05,494.40] vol=1.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-05-26 09:50:00 | 493.70 | 491.62 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:15:00 | 487.30 | 491.62 | 0.00 | ORB-short ORB[491.00,495.50] vol=1.8x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 10:45:00 | 484.84 | 489.51 | 0.00 | T1 1.5R @ 484.84 |
| Target hit | 2025-05-29 15:20:00 | 478.45 | 483.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-06-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:20:00 | 467.00 | 467.50 | 0.00 | ORB-short ORB[468.00,472.00] vol=2.5x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:35:00 | 465.20 | 467.38 | 0.00 | T1 1.5R @ 465.20 |
| Stop hit — per-position SL triggered | 2025-06-03 11:20:00 | 467.00 | 466.84 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:15:00 | 462.85 | 464.15 | 0.00 | ORB-short ORB[463.80,466.90] vol=1.5x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 12:25:00 | 460.90 | 463.35 | 0.00 | T1 1.5R @ 460.90 |
| Target hit | 2025-06-04 15:20:00 | 457.85 | 461.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 464.80 | 463.20 | 0.00 | ORB-long ORB[460.05,463.85] vol=2.2x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-06-05 09:45:00 | 463.34 | 463.88 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:20:00 | 466.70 | 463.99 | 0.00 | ORB-long ORB[460.00,466.00] vol=2.8x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 10:40:00 | 468.90 | 465.23 | 0.00 | T1 1.5R @ 468.90 |
| Target hit | 2025-06-06 15:20:00 | 471.80 | 470.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-06-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 10:20:00 | 469.40 | 470.56 | 0.00 | ORB-short ORB[469.50,475.35] vol=2.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-06-10 11:25:00 | 470.50 | 470.36 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:45:00 | 459.25 | 462.30 | 0.00 | ORB-short ORB[461.55,465.90] vol=2.5x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-06-12 09:50:00 | 460.64 | 462.07 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 447.45 | 445.43 | 0.00 | ORB-long ORB[443.00,446.20] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-06-19 10:45:00 | 446.17 | 446.74 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:55:00 | 442.70 | 444.71 | 0.00 | ORB-short ORB[443.00,447.50] vol=1.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-06-20 10:00:00 | 444.11 | 444.68 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:30:00 | 449.85 | 448.26 | 0.00 | ORB-long ORB[444.25,449.60] vol=1.5x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-06-25 10:50:00 | 448.50 | 448.39 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 446.85 | 449.46 | 0.00 | ORB-short ORB[448.85,451.40] vol=2.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:25:00 | 445.52 | 448.76 | 0.00 | T1 1.5R @ 445.52 |
| Stop hit — per-position SL triggered | 2025-07-01 11:35:00 | 446.85 | 448.72 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 10:25:00 | 455.00 | 457.12 | 0.00 | ORB-short ORB[457.00,461.40] vol=1.7x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:20:00 | 452.91 | 456.11 | 0.00 | T1 1.5R @ 452.91 |
| Stop hit — per-position SL triggered | 2025-07-14 12:30:00 | 455.00 | 455.36 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:00:00 | 448.65 | 451.63 | 0.00 | ORB-short ORB[449.00,454.95] vol=1.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-07-22 12:25:00 | 449.59 | 451.06 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:25:00 | 453.20 | 450.85 | 0.00 | ORB-long ORB[448.55,452.95] vol=1.6x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-07-23 10:30:00 | 451.97 | 450.84 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 459.10 | 457.32 | 0.00 | ORB-long ORB[455.70,458.75] vol=1.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:45:00 | 460.81 | 458.52 | 0.00 | T1 1.5R @ 460.81 |
| Stop hit — per-position SL triggered | 2025-07-24 09:55:00 | 459.10 | 458.79 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:30:00 | 460.05 | 457.44 | 0.00 | ORB-long ORB[452.10,458.70] vol=1.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-07-28 09:55:00 | 458.57 | 458.18 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:10:00 | 456.00 | 454.21 | 0.00 | ORB-long ORB[449.05,454.35] vol=3.1x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-07-29 10:25:00 | 454.32 | 454.48 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:40:00 | 457.60 | 459.38 | 0.00 | ORB-short ORB[457.95,463.25] vol=3.3x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 09:50:00 | 455.44 | 459.13 | 0.00 | T1 1.5R @ 455.44 |
| Stop hit — per-position SL triggered | 2025-07-30 10:20:00 | 457.60 | 458.43 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 457.20 | 455.10 | 0.00 | ORB-long ORB[451.45,456.85] vol=2.6x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:30:00 | 458.95 | 455.82 | 0.00 | T1 1.5R @ 458.95 |
| Stop hit — per-position SL triggered | 2025-07-31 12:00:00 | 457.20 | 456.04 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:35:00 | 438.15 | 439.67 | 0.00 | ORB-short ORB[439.85,443.85] vol=2.8x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:40:00 | 436.28 | 439.01 | 0.00 | T1 1.5R @ 436.28 |
| Target hit | 2025-08-06 13:25:00 | 437.70 | 437.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2025-08-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 11:10:00 | 467.30 | 464.72 | 0.00 | ORB-long ORB[462.55,467.00] vol=1.7x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 12:10:00 | 469.08 | 465.81 | 0.00 | T1 1.5R @ 469.08 |
| Stop hit — per-position SL triggered | 2025-08-20 13:20:00 | 467.30 | 466.59 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 10:15:00 | 462.15 | 463.81 | 0.00 | ORB-short ORB[463.50,467.95] vol=1.7x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-08-21 10:50:00 | 463.10 | 463.12 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 460.15 | 462.26 | 0.00 | ORB-short ORB[461.50,465.20] vol=1.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 09:50:00 | 458.37 | 460.85 | 0.00 | T1 1.5R @ 458.37 |
| Stop hit — per-position SL triggered | 2025-08-22 10:40:00 | 460.15 | 459.60 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:45:00 | 469.95 | 466.93 | 0.00 | ORB-long ORB[463.00,467.25] vol=4.3x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-08-25 09:50:00 | 468.29 | 467.38 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 11:15:00 | 460.10 | 457.28 | 0.00 | ORB-long ORB[454.95,458.45] vol=2.1x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:40:00 | 462.13 | 459.25 | 0.00 | T1 1.5R @ 462.13 |
| Stop hit — per-position SL triggered | 2025-08-28 14:45:00 | 460.10 | 459.27 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:35:00 | 468.25 | 466.32 | 0.00 | ORB-long ORB[463.25,468.00] vol=2.0x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 10:45:00 | 469.92 | 468.07 | 0.00 | T1 1.5R @ 469.92 |
| Stop hit — per-position SL triggered | 2025-09-01 11:00:00 | 468.25 | 468.17 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 09:40:00 | 495.10 | 492.79 | 0.00 | ORB-long ORB[489.00,494.40] vol=2.6x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-09-04 09:45:00 | 493.55 | 493.49 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:40:00 | 486.35 | 484.53 | 0.00 | ORB-long ORB[481.75,486.10] vol=2.1x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:55:00 | 488.33 | 485.50 | 0.00 | T1 1.5R @ 488.33 |
| Stop hit — per-position SL triggered | 2025-09-08 10:40:00 | 486.35 | 486.75 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 10:00:00 | 476.45 | 478.35 | 0.00 | ORB-short ORB[477.25,483.90] vol=2.4x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-09-10 10:05:00 | 477.83 | 478.12 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:45:00 | 487.05 | 485.17 | 0.00 | ORB-long ORB[479.75,486.90] vol=1.6x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-09-16 09:50:00 | 485.53 | 485.24 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 480.35 | 482.38 | 0.00 | ORB-short ORB[481.60,485.05] vol=1.8x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:00:00 | 478.70 | 481.56 | 0.00 | T1 1.5R @ 478.70 |
| Stop hit — per-position SL triggered | 2025-09-19 10:05:00 | 480.35 | 481.38 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:55:00 | 487.20 | 485.56 | 0.00 | ORB-long ORB[481.35,487.00] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 12:00:00 | 488.94 | 486.34 | 0.00 | T1 1.5R @ 488.94 |
| Stop hit — per-position SL triggered | 2025-09-22 13:40:00 | 487.20 | 487.53 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:35:00 | 487.50 | 489.87 | 0.00 | ORB-short ORB[488.00,492.00] vol=2.0x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:45:00 | 485.63 | 489.49 | 0.00 | T1 1.5R @ 485.63 |
| Stop hit — per-position SL triggered | 2025-09-24 12:50:00 | 487.50 | 487.71 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:45:00 | 472.85 | 470.40 | 0.00 | ORB-long ORB[465.55,470.40] vol=1.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-09-30 10:50:00 | 471.36 | 471.06 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:35:00 | 476.00 | 473.34 | 0.00 | ORB-long ORB[472.00,475.00] vol=2.8x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:45:00 | 478.39 | 474.92 | 0.00 | T1 1.5R @ 478.39 |
| Stop hit — per-position SL triggered | 2025-10-06 11:00:00 | 476.00 | 475.28 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:45:00 | 478.65 | 476.10 | 0.00 | ORB-long ORB[472.90,475.95] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-10-07 09:50:00 | 477.47 | 476.25 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 11:10:00 | 481.85 | 485.98 | 0.00 | ORB-short ORB[485.35,491.50] vol=2.3x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:50:00 | 480.17 | 485.42 | 0.00 | T1 1.5R @ 480.17 |
| Stop hit — per-position SL triggered | 2025-10-09 12:50:00 | 481.85 | 484.90 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:00:00 | 479.20 | 482.33 | 0.00 | ORB-short ORB[481.80,488.00] vol=2.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 480.29 | 482.12 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:35:00 | 487.55 | 485.72 | 0.00 | ORB-long ORB[483.00,486.05] vol=2.3x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-10-14 09:45:00 | 486.33 | 485.96 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:35:00 | 489.20 | 487.33 | 0.00 | ORB-long ORB[484.70,487.90] vol=1.9x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-10-16 10:30:00 | 487.90 | 488.74 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:50:00 | 502.95 | 496.36 | 0.00 | ORB-long ORB[488.75,493.40] vol=1.6x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-10-20 09:55:00 | 500.62 | 498.05 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:40:00 | 507.85 | 506.12 | 0.00 | ORB-long ORB[503.00,506.25] vol=1.6x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-10-31 09:45:00 | 506.56 | 506.16 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:35:00 | 517.30 | 515.35 | 0.00 | ORB-long ORB[509.90,516.70] vol=2.0x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:40:00 | 520.17 | 516.62 | 0.00 | T1 1.5R @ 520.17 |
| Target hit | 2025-11-04 12:35:00 | 523.15 | 523.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:40:00 | 498.90 | 503.35 | 0.00 | ORB-short ORB[502.25,508.15] vol=3.2x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-11-07 10:35:00 | 500.65 | 501.41 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:40:00 | 520.75 | 518.90 | 0.00 | ORB-long ORB[515.40,519.70] vol=1.6x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:45:00 | 523.47 | 519.85 | 0.00 | T1 1.5R @ 523.47 |
| Stop hit — per-position SL triggered | 2025-11-10 09:50:00 | 520.75 | 520.00 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:50:00 | 528.35 | 525.63 | 0.00 | ORB-long ORB[520.90,524.85] vol=7.7x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-11-12 09:55:00 | 526.61 | 526.12 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:35:00 | 522.80 | 519.31 | 0.00 | ORB-long ORB[516.40,519.80] vol=2.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-11-21 09:40:00 | 521.34 | 520.09 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:40:00 | 519.40 | 517.62 | 0.00 | ORB-long ORB[511.55,518.90] vol=2.2x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-11-24 10:40:00 | 517.88 | 519.02 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:55:00 | 520.50 | 517.72 | 0.00 | ORB-long ORB[514.75,519.50] vol=1.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 519.02 | 518.81 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:30:00 | 521.65 | 518.16 | 0.00 | ORB-long ORB[514.40,518.50] vol=2.4x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 09:35:00 | 523.59 | 519.54 | 0.00 | T1 1.5R @ 523.59 |
| Stop hit — per-position SL triggered | 2025-12-01 09:40:00 | 521.65 | 519.80 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:20:00 | 524.40 | 527.96 | 0.00 | ORB-short ORB[529.05,536.20] vol=1.9x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 14:40:00 | 522.04 | 525.99 | 0.00 | T1 1.5R @ 522.04 |
| Target hit | 2025-12-05 15:20:00 | 523.20 | 524.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 09:30:00 | 530.45 | 526.78 | 0.00 | ORB-long ORB[519.00,526.50] vol=5.9x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-12-09 09:35:00 | 528.41 | 527.21 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:35:00 | 523.05 | 526.00 | 0.00 | ORB-short ORB[524.35,527.70] vol=1.9x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:50:00 | 521.11 | 525.39 | 0.00 | T1 1.5R @ 521.11 |
| Target hit | 2025-12-10 15:20:00 | 520.45 | 522.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-12-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:55:00 | 515.00 | 518.15 | 0.00 | ORB-short ORB[517.75,523.00] vol=2.5x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-12-11 10:05:00 | 516.37 | 517.30 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:35:00 | 517.55 | 515.94 | 0.00 | ORB-long ORB[513.90,516.50] vol=2.1x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-12-12 10:00:00 | 516.50 | 516.24 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:40:00 | 512.25 | 510.36 | 0.00 | ORB-long ORB[506.30,511.20] vol=3.1x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-12-16 09:50:00 | 511.03 | 510.42 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:55:00 | 501.00 | 504.93 | 0.00 | ORB-short ORB[506.45,511.00] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-12-17 11:00:00 | 501.91 | 504.83 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:45:00 | 502.05 | 497.50 | 0.00 | ORB-long ORB[491.00,497.90] vol=1.9x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-12-19 10:00:00 | 500.15 | 499.12 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 09:50:00 | 507.00 | 508.59 | 0.00 | ORB-short ORB[508.45,512.10] vol=2.6x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-12-23 10:00:00 | 508.35 | 508.26 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 508.25 | 509.58 | 0.00 | ORB-short ORB[509.00,513.95] vol=2.7x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-12-24 09:45:00 | 509.33 | 509.03 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:40:00 | 505.00 | 507.13 | 0.00 | ORB-short ORB[508.50,511.70] vol=4.8x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-12-26 09:50:00 | 506.26 | 506.84 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:50:00 | 493.50 | 496.93 | 0.00 | ORB-short ORB[496.75,502.50] vol=2.2x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-01-01 11:40:00 | 494.49 | 495.59 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:55:00 | 499.00 | 496.34 | 0.00 | ORB-long ORB[492.00,497.25] vol=2.0x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-01-02 13:55:00 | 497.93 | 497.32 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:40:00 | 506.65 | 508.67 | 0.00 | ORB-short ORB[507.30,512.70] vol=1.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-01-06 10:05:00 | 508.13 | 508.25 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:35:00 | 524.20 | 520.40 | 0.00 | ORB-long ORB[516.75,520.10] vol=2.6x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:40:00 | 526.18 | 525.63 | 0.00 | T1 1.5R @ 526.18 |
| Stop hit — per-position SL triggered | 2026-01-08 09:45:00 | 524.20 | 525.57 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:45:00 | 510.30 | 506.96 | 0.00 | ORB-long ORB[504.25,508.65] vol=2.1x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:05:00 | 512.50 | 508.24 | 0.00 | T1 1.5R @ 512.50 |
| Stop hit — per-position SL triggered | 2026-01-19 12:25:00 | 510.30 | 510.31 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:25:00 | 508.40 | 504.52 | 0.00 | ORB-long ORB[502.30,506.30] vol=3.5x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-01-23 11:00:00 | 506.95 | 506.32 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 491.60 | 490.06 | 0.00 | ORB-long ORB[487.75,491.40] vol=2.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 490.68 | 490.10 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:05:00 | 512.75 | 508.68 | 0.00 | ORB-long ORB[504.30,509.50] vol=1.5x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-02-09 10:10:00 | 510.75 | 508.97 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 493.15 | 496.66 | 0.00 | ORB-short ORB[496.75,500.50] vol=7.3x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-02-13 11:45:00 | 494.61 | 495.87 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:40:00 | 484.60 | 487.82 | 0.00 | ORB-short ORB[488.00,492.00] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:50:00 | 482.63 | 486.92 | 0.00 | T1 1.5R @ 482.63 |
| Stop hit — per-position SL triggered | 2026-02-16 09:55:00 | 484.60 | 486.77 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 481.65 | 480.07 | 0.00 | ORB-long ORB[476.10,481.50] vol=1.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 480.59 | 480.15 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 458.65 | 457.62 | 0.00 | ORB-long ORB[453.95,456.60] vol=1.9x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:10:00 | 460.39 | 457.84 | 0.00 | T1 1.5R @ 460.39 |
| Target hit | 2026-02-25 12:30:00 | 460.40 | 460.64 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-23 09:30:00 | 495.70 | 2025-05-23 09:45:00 | 497.59 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-05-23 09:30:00 | 495.70 | 2025-05-23 10:05:00 | 495.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-26 09:45:00 | 495.25 | 2025-05-26 09:50:00 | 493.70 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-29 10:15:00 | 487.30 | 2025-05-29 10:45:00 | 484.84 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-05-29 10:15:00 | 487.30 | 2025-05-29 15:20:00 | 478.45 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2025-06-03 10:20:00 | 467.00 | 2025-06-03 10:35:00 | 465.20 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-03 10:20:00 | 467.00 | 2025-06-03 11:20:00 | 467.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 10:15:00 | 462.85 | 2025-06-04 12:25:00 | 460.90 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-06-04 10:15:00 | 462.85 | 2025-06-04 15:20:00 | 457.85 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2025-06-05 09:30:00 | 464.80 | 2025-06-05 09:45:00 | 463.34 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-06 10:20:00 | 466.70 | 2025-06-06 10:40:00 | 468.90 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-06 10:20:00 | 466.70 | 2025-06-06 15:20:00 | 471.80 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2025-06-10 10:20:00 | 469.40 | 2025-06-10 11:25:00 | 470.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-12 09:45:00 | 459.25 | 2025-06-12 09:50:00 | 460.64 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-19 09:30:00 | 447.45 | 2025-06-19 10:45:00 | 446.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-20 09:55:00 | 442.70 | 2025-06-20 10:00:00 | 444.11 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-25 10:30:00 | 449.85 | 2025-06-25 10:50:00 | 448.50 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-01 10:55:00 | 446.85 | 2025-07-01 11:25:00 | 445.52 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-01 10:55:00 | 446.85 | 2025-07-01 11:35:00 | 446.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-14 10:25:00 | 455.00 | 2025-07-14 11:20:00 | 452.91 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-07-14 10:25:00 | 455.00 | 2025-07-14 12:30:00 | 455.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 11:00:00 | 448.65 | 2025-07-22 12:25:00 | 449.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-23 10:25:00 | 453.20 | 2025-07-23 10:30:00 | 451.97 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-24 09:35:00 | 459.10 | 2025-07-24 09:45:00 | 460.81 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-24 09:35:00 | 459.10 | 2025-07-24 09:55:00 | 459.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-28 09:30:00 | 460.05 | 2025-07-28 09:55:00 | 458.57 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-29 10:10:00 | 456.00 | 2025-07-29 10:25:00 | 454.32 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-30 09:40:00 | 457.60 | 2025-07-30 09:50:00 | 455.44 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-30 09:40:00 | 457.60 | 2025-07-30 10:20:00 | 457.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-31 11:00:00 | 457.20 | 2025-07-31 11:30:00 | 458.95 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-07-31 11:00:00 | 457.20 | 2025-07-31 12:00:00 | 457.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:35:00 | 438.15 | 2025-08-06 10:40:00 | 436.28 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-08-06 10:35:00 | 438.15 | 2025-08-06 13:25:00 | 437.70 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-08-20 11:10:00 | 467.30 | 2025-08-20 12:10:00 | 469.08 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-08-20 11:10:00 | 467.30 | 2025-08-20 13:20:00 | 467.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-21 10:15:00 | 462.15 | 2025-08-21 10:50:00 | 463.10 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-22 09:30:00 | 460.15 | 2025-08-22 09:50:00 | 458.37 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-08-22 09:30:00 | 460.15 | 2025-08-22 10:40:00 | 460.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-25 09:45:00 | 469.95 | 2025-08-25 09:50:00 | 468.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-28 11:15:00 | 460.10 | 2025-08-28 14:40:00 | 462.13 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-08-28 11:15:00 | 460.10 | 2025-08-28 14:45:00 | 460.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 10:35:00 | 468.25 | 2025-09-01 10:45:00 | 469.92 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-01 10:35:00 | 468.25 | 2025-09-01 11:00:00 | 468.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-04 09:40:00 | 495.10 | 2025-09-04 09:45:00 | 493.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-08 09:40:00 | 486.35 | 2025-09-08 09:55:00 | 488.33 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-08 09:40:00 | 486.35 | 2025-09-08 10:40:00 | 486.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-10 10:00:00 | 476.45 | 2025-09-10 10:05:00 | 477.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-16 09:45:00 | 487.05 | 2025-09-16 09:50:00 | 485.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-19 09:55:00 | 480.35 | 2025-09-19 10:00:00 | 478.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-19 09:55:00 | 480.35 | 2025-09-19 10:05:00 | 480.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 10:55:00 | 487.20 | 2025-09-22 12:00:00 | 488.94 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-22 10:55:00 | 487.20 | 2025-09-22 13:40:00 | 487.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 10:35:00 | 487.50 | 2025-09-24 10:45:00 | 485.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-24 10:35:00 | 487.50 | 2025-09-24 12:50:00 | 487.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-30 10:45:00 | 472.85 | 2025-09-30 10:50:00 | 471.36 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-06 10:35:00 | 476.00 | 2025-10-06 10:45:00 | 478.39 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-06 10:35:00 | 476.00 | 2025-10-06 11:00:00 | 476.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-07 09:45:00 | 478.65 | 2025-10-07 09:50:00 | 477.47 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-09 11:10:00 | 481.85 | 2025-10-09 11:50:00 | 480.17 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-09 11:10:00 | 481.85 | 2025-10-09 12:50:00 | 481.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-13 11:00:00 | 479.20 | 2025-10-13 11:15:00 | 480.29 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-14 09:35:00 | 487.55 | 2025-10-14 09:45:00 | 486.33 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-16 09:35:00 | 489.20 | 2025-10-16 10:30:00 | 487.90 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-20 09:50:00 | 502.95 | 2025-10-20 09:55:00 | 500.62 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-10-31 09:40:00 | 507.85 | 2025-10-31 09:45:00 | 506.56 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-04 09:35:00 | 517.30 | 2025-11-04 09:40:00 | 520.17 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-11-04 09:35:00 | 517.30 | 2025-11-04 12:35:00 | 523.15 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2025-11-07 09:40:00 | 498.90 | 2025-11-07 10:35:00 | 500.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-10 09:40:00 | 520.75 | 2025-11-10 09:45:00 | 523.47 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-10 09:40:00 | 520.75 | 2025-11-10 09:50:00 | 520.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 09:50:00 | 528.35 | 2025-11-12 09:55:00 | 526.61 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-21 09:35:00 | 522.80 | 2025-11-21 09:40:00 | 521.34 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-24 09:40:00 | 519.40 | 2025-11-24 10:40:00 | 517.88 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-28 09:55:00 | 520.50 | 2025-11-28 10:15:00 | 519.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-01 09:30:00 | 521.65 | 2025-12-01 09:35:00 | 523.59 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-01 09:30:00 | 521.65 | 2025-12-01 09:40:00 | 521.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 10:20:00 | 524.40 | 2025-12-05 14:40:00 | 522.04 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-12-05 10:20:00 | 524.40 | 2025-12-05 15:20:00 | 523.20 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2025-12-09 09:30:00 | 530.45 | 2025-12-09 09:35:00 | 528.41 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-12-10 10:35:00 | 523.05 | 2025-12-10 10:50:00 | 521.11 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-10 10:35:00 | 523.05 | 2025-12-10 15:20:00 | 520.45 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2025-12-11 09:55:00 | 515.00 | 2025-12-11 10:05:00 | 516.37 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-12 09:35:00 | 517.55 | 2025-12-12 10:00:00 | 516.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-16 09:40:00 | 512.25 | 2025-12-16 09:50:00 | 511.03 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-17 10:55:00 | 501.00 | 2025-12-17 11:00:00 | 501.91 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-19 09:45:00 | 502.05 | 2025-12-19 10:00:00 | 500.15 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-12-23 09:50:00 | 507.00 | 2025-12-23 10:00:00 | 508.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-24 09:30:00 | 508.25 | 2025-12-24 09:45:00 | 509.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-26 09:40:00 | 505.00 | 2025-12-26 09:50:00 | 506.26 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-01 10:50:00 | 493.50 | 2026-01-01 11:40:00 | 494.49 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-02 10:55:00 | 499.00 | 2026-01-02 13:55:00 | 497.93 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-06 09:40:00 | 506.65 | 2026-01-06 10:05:00 | 508.13 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-08 09:35:00 | 524.20 | 2026-01-08 09:40:00 | 526.18 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-08 09:35:00 | 524.20 | 2026-01-08 09:45:00 | 524.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 10:45:00 | 510.30 | 2026-01-19 11:05:00 | 512.50 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-01-19 10:45:00 | 510.30 | 2026-01-19 12:25:00 | 510.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 10:25:00 | 508.40 | 2026-01-23 11:00:00 | 506.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-01 11:05:00 | 491.60 | 2026-02-01 11:15:00 | 490.68 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-09 10:05:00 | 512.75 | 2026-02-09 10:10:00 | 510.75 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-13 11:15:00 | 493.15 | 2026-02-13 11:45:00 | 494.61 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-16 09:40:00 | 484.60 | 2026-02-16 09:50:00 | 482.63 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-16 09:40:00 | 484.60 | 2026-02-16 09:55:00 | 484.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 11:10:00 | 481.65 | 2026-02-18 11:15:00 | 480.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-25 11:00:00 | 458.65 | 2026-02-25 11:10:00 | 460.39 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-25 11:00:00 | 458.65 | 2026-02-25 12:30:00 | 460.40 | TARGET_HIT | 0.50 | 0.38% |
