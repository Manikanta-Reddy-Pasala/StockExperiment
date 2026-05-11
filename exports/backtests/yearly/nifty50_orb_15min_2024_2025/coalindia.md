# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-04-04 15:25:00 (15258 bars)
- **Last close:** 385.50
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
| ENTRY1 | 71 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 8 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 63
- **Target hits / Stop hits / Partials:** 8 / 63 / 33
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 10.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 17 | 36.2% | 4 | 30 | 13 | 0.03% | 1.5% |
| BUY @ 2nd Alert (retest1) | 47 | 17 | 36.2% | 4 | 30 | 13 | 0.03% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 24 | 42.1% | 4 | 33 | 20 | 0.16% | 9.1% |
| SELL @ 2nd Alert (retest1) | 57 | 24 | 42.1% | 4 | 33 | 20 | 0.16% | 9.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 41 | 39.4% | 8 | 63 | 33 | 0.10% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 472.05 | 469.54 | 0.00 | ORB-long ORB[465.25,470.95] vol=2.0x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-05-16 09:40:00 | 470.30 | 470.16 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:55:00 | 498.20 | 493.51 | 0.00 | ORB-long ORB[489.15,496.25] vol=1.8x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 11:10:00 | 500.64 | 494.31 | 0.00 | T1 1.5R @ 500.64 |
| Target hit | 2024-05-24 15:20:00 | 500.85 | 498.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 494.00 | 496.91 | 0.00 | ORB-short ORB[496.00,503.00] vol=2.4x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:15:00 | 491.03 | 495.35 | 0.00 | T1 1.5R @ 491.03 |
| Stop hit — per-position SL triggered | 2024-05-27 11:15:00 | 494.00 | 494.56 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:15:00 | 491.75 | 492.48 | 0.00 | ORB-short ORB[492.60,496.20] vol=1.9x ATR=1.37 |
| Stop hit — per-position SL triggered | 2024-05-28 10:50:00 | 493.12 | 492.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:30:00 | 489.80 | 486.76 | 0.00 | ORB-long ORB[482.10,489.15] vol=2.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-05-29 09:35:00 | 488.16 | 487.06 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:00:00 | 487.65 | 485.54 | 0.00 | ORB-long ORB[479.35,486.50] vol=1.8x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:25:00 | 490.08 | 486.98 | 0.00 | T1 1.5R @ 490.08 |
| Target hit | 2024-06-12 15:00:00 | 489.05 | 489.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2024-06-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:45:00 | 490.30 | 488.43 | 0.00 | ORB-long ORB[485.25,489.60] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-06-14 09:55:00 | 488.81 | 488.71 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 11:00:00 | 474.90 | 475.17 | 0.00 | ORB-short ORB[475.05,480.05] vol=1.5x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-06-24 11:25:00 | 476.02 | 475.20 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 470.55 | 474.26 | 0.00 | ORB-short ORB[474.95,477.95] vol=2.4x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:15:00 | 468.88 | 473.69 | 0.00 | T1 1.5R @ 468.88 |
| Stop hit — per-position SL triggered | 2024-06-25 13:50:00 | 470.55 | 469.98 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:30:00 | 464.85 | 466.49 | 0.00 | ORB-short ORB[466.05,469.00] vol=1.9x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:45:00 | 463.32 | 466.06 | 0.00 | T1 1.5R @ 463.32 |
| Stop hit — per-position SL triggered | 2024-06-27 11:05:00 | 464.85 | 465.81 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 482.20 | 480.07 | 0.00 | ORB-long ORB[476.00,480.55] vol=4.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 480.69 | 480.66 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:25:00 | 485.50 | 489.78 | 0.00 | ORB-short ORB[489.50,494.90] vol=1.7x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 482.95 | 488.58 | 0.00 | T1 1.5R @ 482.95 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 485.50 | 488.05 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:55:00 | 508.35 | 505.69 | 0.00 | ORB-long ORB[498.00,505.60] vol=3.1x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:00:00 | 510.53 | 506.73 | 0.00 | T1 1.5R @ 510.53 |
| Target hit | 2024-07-16 14:05:00 | 511.45 | 512.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 492.40 | 489.25 | 0.00 | ORB-long ORB[485.10,490.25] vol=1.6x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-07-24 09:35:00 | 490.08 | 489.46 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:45:00 | 510.05 | 504.83 | 0.00 | ORB-long ORB[498.25,504.40] vol=2.7x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-07-26 11:30:00 | 508.19 | 506.17 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 10:55:00 | 510.65 | 512.30 | 0.00 | ORB-short ORB[511.10,515.70] vol=1.5x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 12:15:00 | 508.58 | 511.76 | 0.00 | T1 1.5R @ 508.58 |
| Stop hit — per-position SL triggered | 2024-07-29 12:20:00 | 510.65 | 511.71 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:30:00 | 524.10 | 525.78 | 0.00 | ORB-short ORB[525.00,527.95] vol=1.8x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-08-13 09:40:00 | 525.39 | 525.56 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:50:00 | 516.15 | 523.35 | 0.00 | ORB-short ORB[521.80,527.90] vol=2.1x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 10:55:00 | 512.94 | 521.33 | 0.00 | T1 1.5R @ 512.94 |
| Target hit | 2024-08-14 15:20:00 | 506.20 | 508.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 11:15:00 | 504.35 | 506.05 | 0.00 | ORB-short ORB[504.65,507.70] vol=1.5x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-08-16 11:55:00 | 505.64 | 505.77 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 526.25 | 523.81 | 0.00 | ORB-long ORB[521.05,524.40] vol=2.2x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:40:00 | 528.27 | 525.49 | 0.00 | T1 1.5R @ 528.27 |
| Stop hit — per-position SL triggered | 2024-08-20 09:45:00 | 526.25 | 525.60 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 529.00 | 526.72 | 0.00 | ORB-long ORB[524.40,527.45] vol=1.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-08-21 09:35:00 | 527.79 | 526.99 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:55:00 | 529.65 | 531.34 | 0.00 | ORB-short ORB[529.80,535.45] vol=2.6x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 12:20:00 | 527.98 | 530.63 | 0.00 | T1 1.5R @ 527.98 |
| Stop hit — per-position SL triggered | 2024-08-22 12:55:00 | 529.65 | 530.26 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:50:00 | 534.20 | 531.77 | 0.00 | ORB-long ORB[528.85,531.95] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:00:00 | 536.42 | 533.48 | 0.00 | T1 1.5R @ 536.42 |
| Target hit | 2024-08-23 15:20:00 | 538.85 | 537.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:50:00 | 542.90 | 540.70 | 0.00 | ORB-long ORB[536.70,542.25] vol=2.4x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-08-26 10:00:00 | 541.43 | 540.95 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 529.25 | 531.91 | 0.00 | ORB-short ORB[531.75,535.55] vol=2.0x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-08-28 09:50:00 | 530.52 | 531.25 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:30:00 | 518.20 | 521.89 | 0.00 | ORB-short ORB[523.05,527.40] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:35:00 | 516.24 | 521.46 | 0.00 | T1 1.5R @ 516.24 |
| Stop hit — per-position SL triggered | 2024-08-29 10:45:00 | 518.20 | 521.13 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:40:00 | 528.80 | 531.74 | 0.00 | ORB-short ORB[529.95,537.00] vol=1.7x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-08-30 11:50:00 | 530.49 | 530.54 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 510.15 | 512.68 | 0.00 | ORB-short ORB[511.00,516.85] vol=2.0x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:35:00 | 507.90 | 510.93 | 0.00 | T1 1.5R @ 507.90 |
| Target hit | 2024-09-04 13:25:00 | 505.40 | 505.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2024-09-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:45:00 | 484.45 | 486.76 | 0.00 | ORB-short ORB[484.80,488.85] vol=1.7x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:15:00 | 482.37 | 485.62 | 0.00 | T1 1.5R @ 482.37 |
| Stop hit — per-position SL triggered | 2024-09-10 10:25:00 | 484.45 | 485.43 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:05:00 | 492.45 | 494.38 | 0.00 | ORB-short ORB[493.35,497.00] vol=2.1x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 493.71 | 494.26 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:10:00 | 489.85 | 490.76 | 0.00 | ORB-short ORB[491.00,494.10] vol=1.9x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:25:00 | 488.25 | 490.48 | 0.00 | T1 1.5R @ 488.25 |
| Stop hit — per-position SL triggered | 2024-09-16 10:55:00 | 489.85 | 490.11 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 486.00 | 487.73 | 0.00 | ORB-short ORB[488.55,491.60] vol=4.1x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:45:00 | 484.09 | 487.10 | 0.00 | T1 1.5R @ 484.09 |
| Target hit | 2024-09-19 15:20:00 | 477.75 | 481.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:10:00 | 496.85 | 495.08 | 0.00 | ORB-long ORB[490.50,494.70] vol=3.3x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:55:00 | 498.56 | 495.71 | 0.00 | T1 1.5R @ 498.56 |
| Stop hit — per-position SL triggered | 2024-09-23 12:50:00 | 496.85 | 496.05 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:40:00 | 482.80 | 490.55 | 0.00 | ORB-short ORB[494.75,500.55] vol=1.6x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:50:00 | 480.22 | 489.08 | 0.00 | T1 1.5R @ 480.22 |
| Stop hit — per-position SL triggered | 2024-10-07 10:55:00 | 482.80 | 488.80 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:05:00 | 500.35 | 496.73 | 0.00 | ORB-long ORB[492.95,496.65] vol=2.0x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:15:00 | 502.25 | 497.78 | 0.00 | T1 1.5R @ 502.25 |
| Stop hit — per-position SL triggered | 2024-10-14 10:30:00 | 500.35 | 498.43 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:40:00 | 490.45 | 487.51 | 0.00 | ORB-long ORB[483.70,489.00] vol=1.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-10-18 11:00:00 | 488.76 | 487.78 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:30:00 | 480.80 | 484.03 | 0.00 | ORB-short ORB[483.40,487.40] vol=1.6x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:45:00 | 478.12 | 483.27 | 0.00 | T1 1.5R @ 478.12 |
| Stop hit — per-position SL triggered | 2024-10-22 10:50:00 | 480.80 | 483.18 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:35:00 | 451.35 | 448.57 | 0.00 | ORB-long ORB[445.00,449.40] vol=1.7x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:50:00 | 453.23 | 449.69 | 0.00 | T1 1.5R @ 453.23 |
| Stop hit — per-position SL triggered | 2024-10-30 10:30:00 | 451.35 | 451.38 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:10:00 | 424.95 | 421.25 | 0.00 | ORB-long ORB[419.35,424.00] vol=1.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-11-11 12:00:00 | 423.61 | 421.82 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 408.50 | 411.70 | 0.00 | ORB-short ORB[410.00,415.50] vol=1.5x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:45:00 | 406.08 | 410.21 | 0.00 | T1 1.5R @ 406.08 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 408.50 | 409.97 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 09:40:00 | 404.55 | 407.16 | 0.00 | ORB-short ORB[407.20,411.95] vol=3.2x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-11-14 10:00:00 | 406.17 | 406.45 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-11-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:25:00 | 421.25 | 418.75 | 0.00 | ORB-long ORB[414.05,420.35] vol=2.1x ATR=1.24 |
| Stop hit — per-position SL triggered | 2024-11-19 10:40:00 | 420.01 | 419.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:45:00 | 426.70 | 424.60 | 0.00 | ORB-long ORB[420.30,425.85] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-11-25 11:00:00 | 425.30 | 424.70 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 422.20 | 420.80 | 0.00 | ORB-long ORB[419.25,421.25] vol=1.6x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:20:00 | 424.15 | 422.04 | 0.00 | T1 1.5R @ 424.15 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 422.20 | 422.18 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:55:00 | 421.85 | 423.24 | 0.00 | ORB-short ORB[423.00,425.50] vol=1.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-12-03 11:05:00 | 422.60 | 423.22 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:35:00 | 424.80 | 423.75 | 0.00 | ORB-long ORB[421.70,424.50] vol=2.3x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-12-04 10:55:00 | 423.88 | 423.92 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 413.25 | 414.50 | 0.00 | ORB-short ORB[413.85,419.00] vol=2.1x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 414.31 | 414.14 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:55:00 | 416.00 | 417.86 | 0.00 | ORB-short ORB[416.45,420.95] vol=1.5x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 12:30:00 | 414.80 | 416.95 | 0.00 | T1 1.5R @ 414.80 |
| Target hit | 2024-12-09 15:20:00 | 413.95 | 415.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2024-12-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:30:00 | 416.45 | 414.28 | 0.00 | ORB-long ORB[411.50,416.20] vol=1.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-12-10 10:35:00 | 415.50 | 414.33 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 413.55 | 415.10 | 0.00 | ORB-short ORB[414.25,417.85] vol=2.1x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 414.41 | 414.46 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:55:00 | 404.15 | 404.83 | 0.00 | ORB-short ORB[404.20,409.20] vol=2.2x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-12-13 11:55:00 | 405.32 | 404.71 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 408.10 | 409.24 | 0.00 | ORB-short ORB[409.70,411.65] vol=2.4x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:25:00 | 406.93 | 409.03 | 0.00 | T1 1.5R @ 406.93 |
| Stop hit — per-position SL triggered | 2024-12-16 10:35:00 | 408.10 | 408.87 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:20:00 | 405.05 | 407.60 | 0.00 | ORB-short ORB[408.30,410.95] vol=1.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-12-17 10:40:00 | 405.80 | 406.99 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:05:00 | 392.80 | 390.25 | 0.00 | ORB-long ORB[387.00,391.90] vol=2.0x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 11:20:00 | 394.34 | 390.51 | 0.00 | T1 1.5R @ 394.34 |
| Stop hit — per-position SL triggered | 2024-12-19 11:25:00 | 392.80 | 390.57 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:45:00 | 385.70 | 383.97 | 0.00 | ORB-long ORB[382.80,384.90] vol=1.5x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:30:00 | 386.87 | 384.93 | 0.00 | T1 1.5R @ 386.87 |
| Stop hit — per-position SL triggered | 2025-01-01 11:40:00 | 385.70 | 385.01 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:00:00 | 384.25 | 388.15 | 0.00 | ORB-short ORB[389.10,394.05] vol=1.7x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 382.64 | 387.80 | 0.00 | T1 1.5R @ 382.64 |
| Stop hit — per-position SL triggered | 2025-01-06 11:25:00 | 384.25 | 387.25 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:05:00 | 376.55 | 377.75 | 0.00 | ORB-short ORB[376.65,380.85] vol=1.8x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-01-08 11:10:00 | 377.59 | 377.72 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:30:00 | 375.45 | 377.13 | 0.00 | ORB-short ORB[376.05,381.55] vol=1.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-01-09 10:40:00 | 376.41 | 377.03 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 10:15:00 | 368.30 | 366.20 | 0.00 | ORB-long ORB[362.20,367.30] vol=1.9x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:55:00 | 370.20 | 366.98 | 0.00 | T1 1.5R @ 370.20 |
| Stop hit — per-position SL triggered | 2025-01-13 11:15:00 | 368.30 | 367.27 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 09:35:00 | 363.45 | 365.59 | 0.00 | ORB-short ORB[364.55,368.05] vol=1.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-01-14 09:45:00 | 364.87 | 365.22 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:50:00 | 380.65 | 377.71 | 0.00 | ORB-long ORB[374.80,379.55] vol=2.5x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-01-16 11:30:00 | 379.53 | 378.10 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:15:00 | 384.10 | 384.19 | 0.00 | ORB-short ORB[386.25,388.90] vol=15.1x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:20:00 | 382.49 | 384.15 | 0.00 | T1 1.5R @ 382.49 |
| Stop hit — per-position SL triggered | 2025-01-21 10:35:00 | 384.10 | 384.07 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:55:00 | 383.10 | 381.04 | 0.00 | ORB-long ORB[376.55,380.70] vol=1.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-01-23 11:50:00 | 381.99 | 381.70 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:05:00 | 376.30 | 374.69 | 0.00 | ORB-long ORB[370.80,373.50] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 375.33 | 374.92 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:35:00 | 389.25 | 384.37 | 0.00 | ORB-long ORB[381.55,386.00] vol=1.7x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-03-07 10:45:00 | 387.87 | 384.69 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:15:00 | 376.35 | 379.12 | 0.00 | ORB-short ORB[379.20,382.85] vol=3.0x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 374.83 | 378.88 | 0.00 | T1 1.5R @ 374.83 |
| Stop hit — per-position SL triggered | 2025-03-12 11:55:00 | 376.35 | 378.10 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:00:00 | 390.40 | 388.57 | 0.00 | ORB-long ORB[386.45,389.20] vol=3.1x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-03-18 11:05:00 | 389.62 | 388.62 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:30:00 | 394.85 | 392.63 | 0.00 | ORB-long ORB[388.80,392.95] vol=1.7x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 11:50:00 | 396.32 | 393.59 | 0.00 | T1 1.5R @ 396.32 |
| Stop hit — per-position SL triggered | 2025-03-19 13:10:00 | 394.85 | 394.21 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-03-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:05:00 | 392.85 | 395.13 | 0.00 | ORB-short ORB[393.75,397.70] vol=1.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 393.96 | 395.03 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:15:00 | 399.00 | 397.67 | 0.00 | ORB-long ORB[392.05,396.95] vol=2.1x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-03-27 12:05:00 | 398.31 | 398.35 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 11:15:00 | 400.00 | 399.97 | 0.00 | ORB-long ORB[395.70,399.00] vol=4.3x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-03-28 11:40:00 | 398.97 | 399.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:35:00 | 472.05 | 2024-05-16 09:40:00 | 470.30 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-05-24 10:55:00 | 498.20 | 2024-05-24 11:10:00 | 500.64 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-05-24 10:55:00 | 498.20 | 2024-05-24 15:20:00 | 500.85 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2024-05-27 09:45:00 | 494.00 | 2024-05-27 10:15:00 | 491.03 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-27 09:45:00 | 494.00 | 2024-05-27 11:15:00 | 494.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 10:15:00 | 491.75 | 2024-05-28 10:50:00 | 493.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-29 09:30:00 | 489.80 | 2024-05-29 09:35:00 | 488.16 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-12 10:00:00 | 487.65 | 2024-06-12 11:25:00 | 490.08 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-06-12 10:00:00 | 487.65 | 2024-06-12 15:00:00 | 489.05 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-06-14 09:45:00 | 490.30 | 2024-06-14 09:55:00 | 488.81 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-24 11:00:00 | 474.90 | 2024-06-24 11:25:00 | 476.02 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-06-25 11:00:00 | 470.55 | 2024-06-25 11:15:00 | 468.88 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-06-25 11:00:00 | 470.55 | 2024-06-25 13:50:00 | 470.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:30:00 | 464.85 | 2024-06-27 10:45:00 | 463.32 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-27 10:30:00 | 464.85 | 2024-06-27 11:05:00 | 464.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 09:30:00 | 482.20 | 2024-07-02 09:40:00 | 480.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-10 10:25:00 | 485.50 | 2024-07-10 10:35:00 | 482.95 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-10 10:25:00 | 485.50 | 2024-07-10 10:45:00 | 485.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 09:55:00 | 508.35 | 2024-07-16 10:00:00 | 510.53 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-16 09:55:00 | 508.35 | 2024-07-16 14:05:00 | 511.45 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-24 09:30:00 | 492.40 | 2024-07-24 09:35:00 | 490.08 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-07-26 10:45:00 | 510.05 | 2024-07-26 11:30:00 | 508.19 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-29 10:55:00 | 510.65 | 2024-07-29 12:15:00 | 508.58 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-29 10:55:00 | 510.65 | 2024-07-29 12:20:00 | 510.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-13 09:30:00 | 524.10 | 2024-08-13 09:40:00 | 525.39 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-14 10:50:00 | 516.15 | 2024-08-14 10:55:00 | 512.94 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-08-14 10:50:00 | 516.15 | 2024-08-14 15:20:00 | 506.20 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2024-08-16 11:15:00 | 504.35 | 2024-08-16 11:55:00 | 505.64 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-20 09:30:00 | 526.25 | 2024-08-20 09:40:00 | 528.27 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-08-20 09:30:00 | 526.25 | 2024-08-20 09:45:00 | 526.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 09:30:00 | 529.00 | 2024-08-21 09:35:00 | 527.79 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-22 10:55:00 | 529.65 | 2024-08-22 12:20:00 | 527.98 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-08-22 10:55:00 | 529.65 | 2024-08-22 12:55:00 | 529.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-23 09:50:00 | 534.20 | 2024-08-23 10:00:00 | 536.42 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-23 09:50:00 | 534.20 | 2024-08-23 15:20:00 | 538.85 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2024-08-26 09:50:00 | 542.90 | 2024-08-26 10:00:00 | 541.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-28 09:30:00 | 529.25 | 2024-08-28 09:50:00 | 530.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-29 10:30:00 | 518.20 | 2024-08-29 10:35:00 | 516.24 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-29 10:30:00 | 518.20 | 2024-08-29 10:45:00 | 518.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 10:40:00 | 528.80 | 2024-08-30 11:50:00 | 530.49 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-04 09:30:00 | 510.15 | 2024-09-04 09:35:00 | 507.90 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-04 09:30:00 | 510.15 | 2024-09-04 13:25:00 | 505.40 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2024-09-10 09:45:00 | 484.45 | 2024-09-10 10:15:00 | 482.37 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-10 09:45:00 | 484.45 | 2024-09-10 10:25:00 | 484.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-13 11:05:00 | 492.45 | 2024-09-13 11:20:00 | 493.71 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-16 10:10:00 | 489.85 | 2024-09-16 10:25:00 | 488.25 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-16 10:10:00 | 489.85 | 2024-09-16 10:55:00 | 489.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:15:00 | 486.00 | 2024-09-19 10:45:00 | 484.09 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-19 10:15:00 | 486.00 | 2024-09-19 15:20:00 | 477.75 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2024-09-23 11:10:00 | 496.85 | 2024-09-23 11:55:00 | 498.56 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-23 11:10:00 | 496.85 | 2024-09-23 12:50:00 | 496.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:40:00 | 482.80 | 2024-10-07 10:50:00 | 480.22 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-10-07 10:40:00 | 482.80 | 2024-10-07 10:55:00 | 482.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 10:05:00 | 500.35 | 2024-10-14 10:15:00 | 502.25 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-10-14 10:05:00 | 500.35 | 2024-10-14 10:30:00 | 500.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-18 10:40:00 | 490.45 | 2024-10-18 11:00:00 | 488.76 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-22 10:30:00 | 480.80 | 2024-10-22 10:45:00 | 478.12 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-10-22 10:30:00 | 480.80 | 2024-10-22 10:50:00 | 480.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-30 09:35:00 | 451.35 | 2024-10-30 09:50:00 | 453.23 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-10-30 09:35:00 | 451.35 | 2024-10-30 10:30:00 | 451.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 11:10:00 | 424.95 | 2024-11-11 12:00:00 | 423.61 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-13 09:30:00 | 408.50 | 2024-11-13 09:45:00 | 406.08 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-13 09:30:00 | 408.50 | 2024-11-13 09:50:00 | 408.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-14 09:40:00 | 404.55 | 2024-11-14 10:00:00 | 406.17 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-11-19 10:25:00 | 421.25 | 2024-11-19 10:40:00 | 420.01 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-25 10:45:00 | 426.70 | 2024-11-25 11:00:00 | 425.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-28 09:30:00 | 422.20 | 2024-11-28 10:20:00 | 424.15 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-11-28 09:30:00 | 422.20 | 2024-11-28 10:30:00 | 422.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-03 10:55:00 | 421.85 | 2024-12-03 11:05:00 | 422.60 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-04 10:35:00 | 424.80 | 2024-12-04 10:55:00 | 423.88 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-05 10:55:00 | 413.25 | 2024-12-05 12:05:00 | 414.31 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-09 10:55:00 | 416.00 | 2024-12-09 12:30:00 | 414.80 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-09 10:55:00 | 416.00 | 2024-12-09 15:20:00 | 413.95 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-12-10 10:30:00 | 416.45 | 2024-12-10 10:35:00 | 415.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-12 09:35:00 | 413.55 | 2024-12-12 09:50:00 | 414.41 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-13 10:55:00 | 404.15 | 2024-12-13 11:55:00 | 405.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-16 10:15:00 | 408.10 | 2024-12-16 10:25:00 | 406.93 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-16 10:15:00 | 408.10 | 2024-12-16 10:35:00 | 408.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:20:00 | 405.05 | 2024-12-17 10:40:00 | 405.80 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-12-19 11:05:00 | 392.80 | 2024-12-19 11:20:00 | 394.34 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-19 11:05:00 | 392.80 | 2024-12-19 11:25:00 | 392.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:45:00 | 385.70 | 2025-01-01 11:30:00 | 386.87 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-01-01 10:45:00 | 385.70 | 2025-01-01 11:40:00 | 385.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 11:00:00 | 384.25 | 2025-01-06 11:10:00 | 382.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-06 11:00:00 | 384.25 | 2025-01-06 11:25:00 | 384.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-08 11:05:00 | 376.55 | 2025-01-08 11:10:00 | 377.59 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-09 10:30:00 | 375.45 | 2025-01-09 10:40:00 | 376.41 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-13 10:15:00 | 368.30 | 2025-01-13 10:55:00 | 370.20 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-13 10:15:00 | 368.30 | 2025-01-13 11:15:00 | 368.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-14 09:35:00 | 363.45 | 2025-01-14 09:45:00 | 364.87 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-16 10:50:00 | 380.65 | 2025-01-16 11:30:00 | 379.53 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-21 10:15:00 | 384.10 | 2025-01-21 10:20:00 | 382.49 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-21 10:15:00 | 384.10 | 2025-01-21 10:35:00 | 384.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 10:55:00 | 383.10 | 2025-01-23 11:50:00 | 381.99 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-29 11:05:00 | 376.30 | 2025-01-29 11:15:00 | 375.33 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-07 10:35:00 | 389.25 | 2025-03-07 10:45:00 | 387.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-03-12 11:15:00 | 376.35 | 2025-03-12 11:25:00 | 374.83 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-03-12 11:15:00 | 376.35 | 2025-03-12 11:55:00 | 376.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 11:00:00 | 390.40 | 2025-03-18 11:05:00 | 389.62 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-03-19 10:30:00 | 394.85 | 2025-03-19 11:50:00 | 396.32 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-03-19 10:30:00 | 394.85 | 2025-03-19 13:10:00 | 394.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-20 10:05:00 | 392.85 | 2025-03-20 10:15:00 | 393.96 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-27 11:15:00 | 399.00 | 2025-03-27 12:05:00 | 398.31 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-03-28 11:15:00 | 400.00 | 2025-03-28 11:40:00 | 398.97 | STOP_HIT | 1.00 | -0.26% |
