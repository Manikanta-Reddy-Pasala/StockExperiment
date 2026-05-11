# Saregama India Ltd (SAREGAMA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 360.00
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 49
- **Target hits / Stop hits / Partials:** 7 / 49 / 19
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 6.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 13 | 36.1% | 3 | 23 | 10 | 0.13% | 4.8% |
| BUY @ 2nd Alert (retest1) | 36 | 13 | 36.1% | 3 | 23 | 10 | 0.13% | 4.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 13 | 33.3% | 4 | 26 | 9 | 0.04% | 1.6% |
| SELL @ 2nd Alert (retest1) | 39 | 13 | 33.3% | 4 | 26 | 9 | 0.04% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 75 | 26 | 34.7% | 7 | 49 | 19 | 0.08% | 6.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:35:00 | 549.15 | 544.79 | 0.00 | ORB-long ORB[540.10,548.10] vol=3.8x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 09:40:00 | 552.84 | 547.86 | 0.00 | T1 1.5R @ 552.84 |
| Stop hit — per-position SL triggered | 2025-05-14 09:45:00 | 549.15 | 547.99 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:35:00 | 553.80 | 550.27 | 0.00 | ORB-long ORB[543.15,550.55] vol=2.4x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:40:00 | 557.12 | 551.71 | 0.00 | T1 1.5R @ 557.12 |
| Stop hit — per-position SL triggered | 2025-05-15 09:45:00 | 553.80 | 551.91 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-21 10:30:00 | 531.20 | 533.88 | 0.00 | ORB-short ORB[533.50,538.90] vol=1.7x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-05-21 11:35:00 | 532.71 | 532.97 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 11:15:00 | 554.00 | 556.53 | 0.00 | ORB-short ORB[555.05,560.00] vol=5.5x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-05-28 12:25:00 | 555.19 | 556.04 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:00:00 | 539.30 | 541.14 | 0.00 | ORB-short ORB[540.10,546.80] vol=2.5x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-05-30 11:05:00 | 540.68 | 541.26 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:25:00 | 544.80 | 541.33 | 0.00 | ORB-long ORB[537.40,542.95] vol=2.3x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-06-02 10:35:00 | 543.26 | 541.99 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:30:00 | 552.55 | 555.68 | 0.00 | ORB-short ORB[555.35,559.00] vol=1.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-06-11 10:35:00 | 553.66 | 555.30 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 533.00 | 537.65 | 0.00 | ORB-short ORB[537.00,542.95] vol=2.4x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-06-16 09:35:00 | 535.24 | 536.81 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:40:00 | 535.00 | 537.22 | 0.00 | ORB-short ORB[536.10,542.15] vol=2.1x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 12:25:00 | 532.47 | 536.28 | 0.00 | T1 1.5R @ 532.47 |
| Target hit | 2025-06-18 15:20:00 | 524.70 | 530.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-07-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:05:00 | 499.80 | 497.43 | 0.00 | ORB-long ORB[493.35,499.40] vol=3.4x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:30:00 | 502.25 | 498.82 | 0.00 | T1 1.5R @ 502.25 |
| Stop hit — per-position SL triggered | 2025-07-03 10:35:00 | 499.80 | 500.51 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:20:00 | 486.50 | 484.73 | 0.00 | ORB-long ORB[483.75,485.85] vol=2.0x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-07-16 11:00:00 | 485.34 | 485.66 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:40:00 | 496.40 | 497.94 | 0.00 | ORB-short ORB[496.75,503.75] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-07-23 10:45:00 | 497.58 | 497.93 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:15:00 | 492.65 | 494.77 | 0.00 | ORB-short ORB[493.00,497.75] vol=1.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:25:00 | 491.23 | 494.52 | 0.00 | T1 1.5R @ 491.23 |
| Stop hit — per-position SL triggered | 2025-07-24 11:35:00 | 492.65 | 494.46 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 11:00:00 | 486.00 | 491.75 | 0.00 | ORB-short ORB[493.15,495.50] vol=1.9x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:45:00 | 483.56 | 490.63 | 0.00 | T1 1.5R @ 483.56 |
| Stop hit — per-position SL triggered | 2025-07-25 14:40:00 | 486.00 | 488.39 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:45:00 | 475.50 | 480.71 | 0.00 | ORB-short ORB[480.70,484.30] vol=2.8x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-07-30 09:50:00 | 477.38 | 478.71 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 09:45:00 | 482.15 | 486.17 | 0.00 | ORB-short ORB[485.15,491.95] vol=2.9x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-08-12 09:55:00 | 484.49 | 485.79 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:50:00 | 478.90 | 481.30 | 0.00 | ORB-short ORB[480.25,486.90] vol=3.7x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:00:00 | 477.10 | 480.59 | 0.00 | T1 1.5R @ 477.10 |
| Stop hit — per-position SL triggered | 2025-08-13 11:40:00 | 478.90 | 479.92 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 481.45 | 485.54 | 0.00 | ORB-short ORB[483.10,489.00] vol=1.8x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-08-18 10:10:00 | 483.27 | 484.42 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 480.85 | 482.37 | 0.00 | ORB-short ORB[481.20,485.00] vol=2.9x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-08-20 09:35:00 | 481.85 | 482.36 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 480.10 | 481.81 | 0.00 | ORB-short ORB[480.25,487.05] vol=3.4x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-08-26 09:50:00 | 481.47 | 481.61 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 09:45:00 | 481.35 | 479.85 | 0.00 | ORB-long ORB[475.30,481.20] vol=1.8x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-08-28 10:05:00 | 478.97 | 479.96 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:40:00 | 488.40 | 484.29 | 0.00 | ORB-long ORB[479.25,484.95] vol=3.7x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-09-01 09:45:00 | 486.51 | 484.71 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 09:30:00 | 499.40 | 496.78 | 0.00 | ORB-long ORB[491.05,498.20] vol=2.4x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-09-04 09:55:00 | 497.56 | 497.25 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:20:00 | 506.50 | 503.90 | 0.00 | ORB-long ORB[497.00,504.45] vol=1.5x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-09-10 10:30:00 | 504.20 | 504.01 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:50:00 | 491.10 | 493.50 | 0.00 | ORB-short ORB[491.65,495.75] vol=2.0x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-09-15 11:45:00 | 492.62 | 491.86 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:45:00 | 483.20 | 485.41 | 0.00 | ORB-short ORB[484.55,488.10] vol=1.8x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:50:00 | 480.88 | 484.64 | 0.00 | T1 1.5R @ 480.88 |
| Target hit | 2025-09-23 13:55:00 | 480.40 | 480.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2025-09-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:00:00 | 472.60 | 475.56 | 0.00 | ORB-short ORB[475.05,478.40] vol=3.1x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 473.91 | 475.10 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 11:05:00 | 467.90 | 463.81 | 0.00 | ORB-long ORB[461.75,467.60] vol=2.2x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 14:25:00 | 470.07 | 465.46 | 0.00 | T1 1.5R @ 470.07 |
| Stop hit — per-position SL triggered | 2025-09-30 14:55:00 | 467.90 | 465.67 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:40:00 | 473.00 | 470.82 | 0.00 | ORB-long ORB[467.10,471.10] vol=10.2x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-10-01 10:45:00 | 471.47 | 471.27 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:10:00 | 463.35 | 464.95 | 0.00 | ORB-short ORB[464.00,467.50] vol=5.1x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-10-07 11:05:00 | 464.53 | 464.68 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:25:00 | 464.35 | 467.73 | 0.00 | ORB-short ORB[466.50,471.20] vol=1.6x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-10-08 10:30:00 | 465.73 | 467.63 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 450.45 | 452.90 | 0.00 | ORB-short ORB[452.05,456.00] vol=3.3x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-10-14 11:25:00 | 451.38 | 452.84 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:25:00 | 451.65 | 453.43 | 0.00 | ORB-short ORB[453.90,459.25] vol=1.6x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:30:00 | 449.66 | 452.26 | 0.00 | T1 1.5R @ 449.66 |
| Target hit | 2025-10-17 11:20:00 | 450.75 | 450.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2025-10-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:00:00 | 455.10 | 453.43 | 0.00 | ORB-long ORB[451.05,453.60] vol=2.4x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:05:00 | 456.88 | 454.60 | 0.00 | T1 1.5R @ 456.88 |
| Target hit | 2025-10-24 14:00:00 | 457.75 | 458.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2025-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:30:00 | 459.05 | 458.18 | 0.00 | ORB-long ORB[455.20,458.90] vol=3.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-11-04 09:45:00 | 457.93 | 458.53 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:35:00 | 419.15 | 414.05 | 0.00 | ORB-long ORB[409.00,414.95] vol=1.6x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-11-10 09:40:00 | 417.41 | 414.89 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:40:00 | 377.00 | 375.38 | 0.00 | ORB-long ORB[372.35,376.25] vol=2.3x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 09:50:00 | 379.13 | 376.36 | 0.00 | T1 1.5R @ 379.13 |
| Stop hit — per-position SL triggered | 2025-12-16 10:25:00 | 377.00 | 376.81 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 353.25 | 350.65 | 0.00 | ORB-long ORB[347.05,352.10] vol=1.8x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-12-31 11:25:00 | 352.42 | 351.07 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:55:00 | 353.60 | 351.63 | 0.00 | ORB-long ORB[348.20,351.95] vol=2.7x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:45:00 | 355.52 | 353.06 | 0.00 | T1 1.5R @ 355.52 |
| Stop hit — per-position SL triggered | 2026-01-02 11:45:00 | 353.60 | 353.27 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-01-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:55:00 | 354.75 | 354.34 | 0.00 | ORB-long ORB[351.00,354.10] vol=7.8x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-01-07 10:05:00 | 353.45 | 354.29 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:35:00 | 351.50 | 354.65 | 0.00 | ORB-short ORB[354.05,357.90] vol=1.9x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 352.99 | 354.39 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:30:00 | 329.40 | 326.76 | 0.00 | ORB-long ORB[324.15,328.60] vol=6.8x ATR=1.85 |
| Stop hit — per-position SL triggered | 2026-01-22 10:00:00 | 327.55 | 327.31 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:15:00 | 334.40 | 329.00 | 0.00 | ORB-long ORB[324.15,326.95] vol=1.9x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:55:00 | 336.98 | 331.69 | 0.00 | T1 1.5R @ 336.98 |
| Stop hit — per-position SL triggered | 2026-01-30 13:05:00 | 334.40 | 333.42 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-02-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 10:25:00 | 332.00 | 329.90 | 0.00 | ORB-long ORB[326.10,330.60] vol=6.8x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-02-05 10:30:00 | 330.39 | 330.13 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 344.15 | 341.56 | 0.00 | ORB-long ORB[338.15,342.10] vol=2.1x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 342.93 | 341.64 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 324.55 | 327.55 | 0.00 | ORB-short ORB[327.60,331.20] vol=8.8x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 325.80 | 326.77 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:55:00 | 322.00 | 319.55 | 0.00 | ORB-long ORB[316.65,319.55] vol=2.4x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:00:00 | 323.94 | 320.38 | 0.00 | T1 1.5R @ 323.94 |
| Target hit | 2026-03-05 15:20:00 | 328.00 | 323.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 332.60 | 330.07 | 0.00 | ORB-long ORB[327.25,331.75] vol=3.5x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 331.30 | 330.34 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-04-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 10:55:00 | 314.30 | 315.08 | 0.00 | ORB-short ORB[316.05,320.00] vol=1.9x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 12:35:00 | 312.88 | 314.80 | 0.00 | T1 1.5R @ 312.88 |
| Target hit | 2026-04-07 15:20:00 | 310.05 | 313.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 342.00 | 343.66 | 0.00 | ORB-short ORB[343.10,346.75] vol=2.3x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 340.01 | 343.28 | 0.00 | T1 1.5R @ 340.01 |
| Stop hit — per-position SL triggered | 2026-04-16 10:50:00 | 342.00 | 341.80 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 339.05 | 340.67 | 0.00 | ORB-short ORB[339.35,343.95] vol=2.1x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:55:00 | 337.16 | 340.16 | 0.00 | T1 1.5R @ 337.16 |
| Stop hit — per-position SL triggered | 2026-04-22 10:00:00 | 339.05 | 340.02 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 351.50 | 349.33 | 0.00 | ORB-long ORB[345.55,349.50] vol=5.4x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 353.64 | 350.32 | 0.00 | T1 1.5R @ 353.64 |
| Target hit | 2026-04-27 15:20:00 | 360.35 | 359.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 359.20 | 360.80 | 0.00 | ORB-short ORB[359.50,364.65] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-04-28 09:40:00 | 360.99 | 360.79 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 348.30 | 346.07 | 0.00 | ORB-long ORB[344.00,346.90] vol=7.9x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 347.23 | 346.90 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 343.30 | 344.94 | 0.00 | ORB-short ORB[344.30,347.10] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 344.15 | 344.77 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 347.95 | 349.73 | 0.00 | ORB-short ORB[348.80,352.45] vol=1.6x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-05-08 09:45:00 | 349.11 | 349.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:35:00 | 549.15 | 2025-05-14 09:40:00 | 552.84 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-05-14 09:35:00 | 549.15 | 2025-05-14 09:45:00 | 549.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 09:35:00 | 553.80 | 2025-05-15 09:40:00 | 557.12 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-05-15 09:35:00 | 553.80 | 2025-05-15 09:45:00 | 553.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-21 10:30:00 | 531.20 | 2025-05-21 11:35:00 | 532.71 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-28 11:15:00 | 554.00 | 2025-05-28 12:25:00 | 555.19 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-30 11:00:00 | 539.30 | 2025-05-30 11:05:00 | 540.68 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-02 10:25:00 | 544.80 | 2025-06-02 10:35:00 | 543.26 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-11 10:30:00 | 552.55 | 2025-06-11 10:35:00 | 553.66 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-06-16 09:30:00 | 533.00 | 2025-06-16 09:35:00 | 535.24 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-06-18 10:40:00 | 535.00 | 2025-06-18 12:25:00 | 532.47 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-06-18 10:40:00 | 535.00 | 2025-06-18 15:20:00 | 524.70 | TARGET_HIT | 0.50 | 1.93% |
| BUY | retest1 | 2025-07-03 10:05:00 | 499.80 | 2025-07-03 10:30:00 | 502.25 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-03 10:05:00 | 499.80 | 2025-07-03 10:35:00 | 499.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 10:20:00 | 486.50 | 2025-07-16 11:00:00 | 485.34 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-23 10:40:00 | 496.40 | 2025-07-23 10:45:00 | 497.58 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-24 11:15:00 | 492.65 | 2025-07-24 11:25:00 | 491.23 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-07-24 11:15:00 | 492.65 | 2025-07-24 11:35:00 | 492.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 11:00:00 | 486.00 | 2025-07-25 11:45:00 | 483.56 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-25 11:00:00 | 486.00 | 2025-07-25 14:40:00 | 486.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-30 09:45:00 | 475.50 | 2025-07-30 09:50:00 | 477.38 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-08-12 09:45:00 | 482.15 | 2025-08-12 09:55:00 | 484.49 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-08-13 10:50:00 | 478.90 | 2025-08-13 11:00:00 | 477.10 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-13 10:50:00 | 478.90 | 2025-08-13 11:40:00 | 478.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-18 09:30:00 | 481.45 | 2025-08-18 10:10:00 | 483.27 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-08-20 09:30:00 | 480.85 | 2025-08-20 09:35:00 | 481.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-26 09:30:00 | 480.10 | 2025-08-26 09:50:00 | 481.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-28 09:45:00 | 481.35 | 2025-08-28 10:05:00 | 478.97 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-09-01 09:40:00 | 488.40 | 2025-09-01 09:45:00 | 486.51 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-04 09:30:00 | 499.40 | 2025-09-04 09:55:00 | 497.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-10 10:20:00 | 506.50 | 2025-09-10 10:30:00 | 504.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-09-15 09:50:00 | 491.10 | 2025-09-15 11:45:00 | 492.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-23 09:45:00 | 483.20 | 2025-09-23 09:50:00 | 480.88 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-09-23 09:45:00 | 483.20 | 2025-09-23 13:55:00 | 480.40 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-09-25 10:00:00 | 472.60 | 2025-09-25 10:15:00 | 473.91 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-30 11:05:00 | 467.90 | 2025-09-30 14:25:00 | 470.07 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-09-30 11:05:00 | 467.90 | 2025-09-30 14:55:00 | 467.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 10:40:00 | 473.00 | 2025-10-01 10:45:00 | 471.47 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-07 10:10:00 | 463.35 | 2025-10-07 11:05:00 | 464.53 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-08 10:25:00 | 464.35 | 2025-10-08 10:30:00 | 465.73 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-14 11:15:00 | 450.45 | 2025-10-14 11:25:00 | 451.38 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-17 10:25:00 | 451.65 | 2025-10-17 10:30:00 | 449.66 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-10-17 10:25:00 | 451.65 | 2025-10-17 11:20:00 | 450.75 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-10-24 10:00:00 | 455.10 | 2025-10-24 10:05:00 | 456.88 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-10-24 10:00:00 | 455.10 | 2025-10-24 14:00:00 | 457.75 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-11-04 09:30:00 | 459.05 | 2025-11-04 09:45:00 | 457.93 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-10 09:35:00 | 419.15 | 2025-11-10 09:40:00 | 417.41 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-12-16 09:40:00 | 377.00 | 2025-12-16 09:50:00 | 379.13 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-12-16 09:40:00 | 377.00 | 2025-12-16 10:25:00 | 377.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 11:00:00 | 353.25 | 2025-12-31 11:25:00 | 352.42 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-02 09:55:00 | 353.60 | 2026-01-02 10:45:00 | 355.52 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-01-02 09:55:00 | 353.60 | 2026-01-02 11:45:00 | 353.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 09:55:00 | 354.75 | 2026-01-07 10:05:00 | 353.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-01-13 09:35:00 | 351.50 | 2026-01-13 09:45:00 | 352.99 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-01-22 09:30:00 | 329.40 | 2026-01-22 10:00:00 | 327.55 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2026-01-30 10:15:00 | 334.40 | 2026-01-30 10:55:00 | 336.98 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-01-30 10:15:00 | 334.40 | 2026-01-30 13:05:00 | 334.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-05 10:25:00 | 332.00 | 2026-02-05 10:30:00 | 330.39 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-02-17 10:25:00 | 344.15 | 2026-02-17 10:30:00 | 342.93 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-25 10:50:00 | 324.55 | 2026-02-25 11:05:00 | 325.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-05 09:55:00 | 322.00 | 2026-03-05 10:00:00 | 323.94 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-05 09:55:00 | 322.00 | 2026-03-05 15:20:00 | 328.00 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2026-03-11 10:10:00 | 332.60 | 2026-03-11 10:15:00 | 331.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-07 10:55:00 | 314.30 | 2026-04-07 12:35:00 | 312.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-07 10:55:00 | 314.30 | 2026-04-07 15:20:00 | 310.05 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2026-04-16 09:35:00 | 342.00 | 2026-04-16 09:40:00 | 340.01 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-16 09:35:00 | 342.00 | 2026-04-16 10:50:00 | 342.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 09:45:00 | 339.05 | 2026-04-22 09:55:00 | 337.16 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-22 09:45:00 | 339.05 | 2026-04-22 10:00:00 | 339.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:45:00 | 351.50 | 2026-04-27 09:50:00 | 353.64 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-27 09:45:00 | 351.50 | 2026-04-27 15:20:00 | 360.35 | TARGET_HIT | 0.50 | 2.52% |
| SELL | retest1 | 2026-04-28 09:35:00 | 359.20 | 2026-04-28 09:40:00 | 360.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-05-04 11:10:00 | 348.30 | 2026-05-04 12:10:00 | 347.23 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-06 10:55:00 | 343.30 | 2026-05-06 11:10:00 | 344.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-08 09:40:00 | 347.95 | 2026-05-08 09:45:00 | 349.11 | STOP_HIT | 1.00 | -0.33% |
