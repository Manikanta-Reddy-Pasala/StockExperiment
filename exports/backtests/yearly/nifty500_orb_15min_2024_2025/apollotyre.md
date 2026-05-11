# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 408.65
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
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 18 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 130 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 78
- **Target hits / Stop hits / Partials:** 18 / 78 / 34
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 15.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 34 | 44.7% | 13 | 42 | 21 | 0.19% | 14.3% |
| BUY @ 2nd Alert (retest1) | 76 | 34 | 44.7% | 13 | 42 | 21 | 0.19% | 14.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 18 | 33.3% | 5 | 36 | 13 | 0.02% | 1.1% |
| SELL @ 2nd Alert (retest1) | 54 | 18 | 33.3% | 5 | 36 | 13 | 0.02% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 130 | 52 | 40.0% | 18 | 78 | 34 | 0.12% | 15.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 483.30 | 487.55 | 0.00 | ORB-short ORB[487.15,493.70] vol=1.9x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-05-23 15:20:00 | 483.50 | 484.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:15:00 | 486.30 | 483.44 | 0.00 | ORB-long ORB[482.10,485.40] vol=3.1x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-05-24 11:25:00 | 485.15 | 483.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 474.75 | 477.76 | 0.00 | ORB-short ORB[476.15,483.00] vol=3.4x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-05-27 09:35:00 | 476.47 | 477.58 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 09:40:00 | 476.45 | 474.35 | 0.00 | ORB-long ORB[469.65,474.90] vol=3.1x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-05-30 09:45:00 | 475.03 | 474.43 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:10:00 | 476.60 | 474.12 | 0.00 | ORB-long ORB[470.10,475.00] vol=2.0x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:15:00 | 478.62 | 475.37 | 0.00 | T1 1.5R @ 478.62 |
| Target hit | 2024-06-07 12:50:00 | 480.75 | 480.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2024-06-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 10:50:00 | 484.70 | 484.98 | 0.00 | ORB-short ORB[485.15,488.20] vol=3.8x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-06-10 11:05:00 | 486.23 | 485.03 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:45:00 | 481.20 | 479.85 | 0.00 | ORB-long ORB[477.10,480.60] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-06-13 09:50:00 | 479.99 | 479.92 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:35:00 | 473.30 | 475.28 | 0.00 | ORB-short ORB[474.20,479.15] vol=1.6x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-06-14 10:00:00 | 474.36 | 474.51 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:40:00 | 483.85 | 481.05 | 0.00 | ORB-long ORB[477.10,480.65] vol=6.1x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:50:00 | 486.04 | 483.04 | 0.00 | T1 1.5R @ 486.04 |
| Target hit | 2024-06-18 15:10:00 | 484.15 | 484.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2024-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:45:00 | 512.35 | 514.33 | 0.00 | ORB-short ORB[513.20,518.70] vol=1.7x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 513.84 | 514.05 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:30:00 | 521.80 | 519.86 | 0.00 | ORB-long ORB[516.55,520.40] vol=2.0x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-06-28 09:50:00 | 519.81 | 520.58 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:15:00 | 540.85 | 544.28 | 0.00 | ORB-short ORB[544.60,550.15] vol=2.1x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-07-02 10:40:00 | 542.71 | 543.58 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 09:40:00 | 533.15 | 535.22 | 0.00 | ORB-short ORB[535.00,539.65] vol=1.6x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:45:00 | 530.52 | 534.07 | 0.00 | T1 1.5R @ 530.52 |
| Stop hit — per-position SL triggered | 2024-07-03 10:50:00 | 533.15 | 534.00 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 528.35 | 526.92 | 0.00 | ORB-long ORB[522.45,527.65] vol=2.8x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:40:00 | 530.81 | 527.55 | 0.00 | T1 1.5R @ 530.81 |
| Stop hit — per-position SL triggered | 2024-07-05 09:50:00 | 528.35 | 528.18 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:10:00 | 528.40 | 525.46 | 0.00 | ORB-long ORB[522.55,527.85] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 526.82 | 525.88 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 545.35 | 543.18 | 0.00 | ORB-long ORB[540.00,545.00] vol=1.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-07-16 09:35:00 | 543.68 | 543.30 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 545.65 | 548.13 | 0.00 | ORB-short ORB[546.60,551.00] vol=1.5x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 547.46 | 548.11 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:55:00 | 528.00 | 524.20 | 0.00 | ORB-long ORB[519.65,526.25] vol=1.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-07-22 11:00:00 | 525.91 | 524.29 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:35:00 | 524.95 | 523.37 | 0.00 | ORB-long ORB[516.35,523.90] vol=4.4x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 09:50:00 | 528.25 | 524.13 | 0.00 | T1 1.5R @ 528.25 |
| Target hit | 2024-07-24 15:20:00 | 538.95 | 533.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:50:00 | 538.35 | 535.66 | 0.00 | ORB-long ORB[531.25,536.00] vol=2.8x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 536.21 | 536.91 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:15:00 | 556.40 | 550.93 | 0.00 | ORB-long ORB[543.65,551.75] vol=2.4x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:20:00 | 559.72 | 552.41 | 0.00 | T1 1.5R @ 559.72 |
| Stop hit — per-position SL triggered | 2024-07-30 10:25:00 | 556.40 | 552.82 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 09:30:00 | 552.65 | 555.14 | 0.00 | ORB-short ORB[553.45,558.20] vol=2.0x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-08-01 09:45:00 | 554.20 | 554.46 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 10:50:00 | 517.00 | 521.67 | 0.00 | ORB-short ORB[520.00,526.95] vol=1.6x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-08-07 11:00:00 | 519.53 | 521.25 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:20:00 | 489.95 | 491.76 | 0.00 | ORB-short ORB[491.10,495.75] vol=2.2x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-08-13 10:55:00 | 491.30 | 491.60 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:15:00 | 481.70 | 486.09 | 0.00 | ORB-short ORB[488.00,491.55] vol=1.6x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-08-16 14:05:00 | 483.20 | 482.77 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:35:00 | 489.30 | 486.52 | 0.00 | ORB-long ORB[483.65,487.95] vol=1.9x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-08-20 10:45:00 | 488.07 | 486.86 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 502.05 | 505.66 | 0.00 | ORB-short ORB[506.20,509.90] vol=3.9x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-08-23 09:35:00 | 503.27 | 505.02 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:50:00 | 511.35 | 506.70 | 0.00 | ORB-long ORB[503.00,505.40] vol=3.5x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-08-27 11:00:00 | 510.03 | 507.25 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:25:00 | 499.95 | 498.92 | 0.00 | ORB-long ORB[495.35,499.80] vol=2.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-08-29 10:35:00 | 499.06 | 498.99 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:10:00 | 514.80 | 511.51 | 0.00 | ORB-long ORB[509.10,512.90] vol=2.3x ATR=1.30 |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 513.50 | 512.08 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 506.20 | 509.69 | 0.00 | ORB-short ORB[508.30,513.90] vol=1.9x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 504.12 | 508.10 | 0.00 | T1 1.5R @ 504.12 |
| Target hit | 2024-09-06 13:25:00 | 505.80 | 505.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — SELL (started 2024-09-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:55:00 | 524.65 | 529.62 | 0.00 | ORB-short ORB[528.00,534.10] vol=2.0x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-09-16 10:00:00 | 526.56 | 529.13 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 519.30 | 522.21 | 0.00 | ORB-short ORB[521.15,527.90] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:55:00 | 517.09 | 521.12 | 0.00 | T1 1.5R @ 517.09 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 519.30 | 520.98 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 517.05 | 515.58 | 0.00 | ORB-long ORB[511.70,516.95] vol=1.9x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 515.36 | 515.76 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:05:00 | 522.15 | 520.11 | 0.00 | ORB-long ORB[518.00,521.85] vol=2.2x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:15:00 | 524.25 | 521.55 | 0.00 | T1 1.5R @ 524.25 |
| Target hit | 2024-09-24 15:20:00 | 529.85 | 525.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 505.85 | 506.08 | 0.00 | ORB-short ORB[510.15,515.00] vol=2.1x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-10-07 11:10:00 | 508.02 | 506.15 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:45:00 | 500.75 | 495.79 | 0.00 | ORB-long ORB[493.00,498.85] vol=2.1x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-10-08 09:55:00 | 497.85 | 496.26 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 11:05:00 | 512.35 | 514.49 | 0.00 | ORB-short ORB[514.00,519.05] vol=11.1x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-10-09 11:20:00 | 514.10 | 514.45 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:15:00 | 502.30 | 505.02 | 0.00 | ORB-short ORB[504.00,508.70] vol=1.7x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-10-11 11:50:00 | 503.29 | 504.68 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:35:00 | 512.50 | 509.54 | 0.00 | ORB-long ORB[505.85,509.55] vol=2.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:40:00 | 514.72 | 510.96 | 0.00 | T1 1.5R @ 514.72 |
| Target hit | 2024-10-15 10:00:00 | 513.50 | 513.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2024-10-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:05:00 | 510.30 | 513.90 | 0.00 | ORB-short ORB[512.55,518.50] vol=2.4x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-10-16 11:25:00 | 511.64 | 513.51 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 10:55:00 | 510.50 | 506.88 | 0.00 | ORB-long ORB[505.20,510.30] vol=2.2x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:00:00 | 513.07 | 507.91 | 0.00 | T1 1.5R @ 513.07 |
| Stop hit — per-position SL triggered | 2024-10-21 11:30:00 | 510.50 | 509.56 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:15:00 | 499.75 | 495.47 | 0.00 | ORB-long ORB[493.00,499.00] vol=2.0x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:15:00 | 504.03 | 497.21 | 0.00 | T1 1.5R @ 504.03 |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 499.75 | 499.60 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:35:00 | 480.20 | 483.22 | 0.00 | ORB-short ORB[486.45,490.55] vol=2.1x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-10-29 10:55:00 | 482.04 | 482.69 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:40:00 | 502.40 | 498.61 | 0.00 | ORB-long ORB[496.00,499.60] vol=1.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-10-30 09:50:00 | 500.37 | 499.05 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:30:00 | 501.25 | 498.90 | 0.00 | ORB-long ORB[494.65,499.85] vol=2.0x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-10-31 09:35:00 | 499.42 | 499.30 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:40:00 | 496.80 | 494.09 | 0.00 | ORB-long ORB[492.00,495.85] vol=1.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-11-07 09:50:00 | 495.16 | 494.48 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 480.55 | 476.86 | 0.00 | ORB-long ORB[471.00,477.75] vol=2.4x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-11-19 09:40:00 | 478.76 | 477.43 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:50:00 | 481.95 | 480.19 | 0.00 | ORB-long ORB[477.15,481.90] vol=1.6x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 11:10:00 | 484.36 | 480.60 | 0.00 | T1 1.5R @ 484.36 |
| Stop hit — per-position SL triggered | 2024-11-21 14:35:00 | 481.95 | 482.45 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:50:00 | 487.40 | 485.27 | 0.00 | ORB-long ORB[479.90,485.85] vol=1.5x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-11-22 11:20:00 | 485.80 | 485.95 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:40:00 | 510.00 | 507.98 | 0.00 | ORB-long ORB[504.35,509.15] vol=1.9x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-11-26 09:55:00 | 508.56 | 508.69 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:55:00 | 508.00 | 510.35 | 0.00 | ORB-short ORB[508.65,514.00] vol=1.5x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:15:00 | 506.22 | 510.01 | 0.00 | T1 1.5R @ 506.22 |
| Stop hit — per-position SL triggered | 2024-11-28 11:30:00 | 508.00 | 509.78 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:10:00 | 504.65 | 507.37 | 0.00 | ORB-short ORB[505.00,509.80] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-11-29 10:25:00 | 506.26 | 507.07 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:40:00 | 517.45 | 515.60 | 0.00 | ORB-long ORB[513.85,516.80] vol=1.8x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-12-03 09:50:00 | 516.22 | 515.80 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:45:00 | 524.50 | 523.00 | 0.00 | ORB-long ORB[518.75,523.95] vol=1.9x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:05:00 | 527.24 | 523.54 | 0.00 | T1 1.5R @ 527.24 |
| Target hit | 2024-12-04 15:20:00 | 534.90 | 528.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2024-12-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:45:00 | 542.00 | 540.31 | 0.00 | ORB-long ORB[534.90,541.90] vol=2.7x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-12-05 10:55:00 | 540.29 | 541.06 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:35:00 | 544.10 | 542.60 | 0.00 | ORB-long ORB[537.20,543.75] vol=1.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 09:45:00 | 546.44 | 543.23 | 0.00 | T1 1.5R @ 546.44 |
| Target hit | 2024-12-09 12:25:00 | 550.50 | 550.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 539.15 | 541.59 | 0.00 | ORB-short ORB[540.05,547.00] vol=2.2x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-12-10 10:10:00 | 540.95 | 540.99 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:55:00 | 542.85 | 539.80 | 0.00 | ORB-long ORB[536.65,541.35] vol=1.5x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-12-12 10:05:00 | 541.52 | 540.28 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 540.00 | 540.44 | 0.00 | ORB-short ORB[540.05,542.20] vol=3.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 541.04 | 540.17 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:40:00 | 536.10 | 537.94 | 0.00 | ORB-short ORB[537.20,540.45] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:15:00 | 534.14 | 537.44 | 0.00 | T1 1.5R @ 534.14 |
| Target hit | 2024-12-17 15:20:00 | 532.55 | 534.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2024-12-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:45:00 | 534.00 | 531.65 | 0.00 | ORB-long ORB[526.15,531.90] vol=1.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-12-20 12:20:00 | 532.51 | 533.11 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 11:10:00 | 535.65 | 534.51 | 0.00 | ORB-long ORB[531.00,535.50] vol=2.2x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:20:00 | 537.43 | 534.80 | 0.00 | T1 1.5R @ 537.43 |
| Stop hit — per-position SL triggered | 2024-12-24 12:20:00 | 535.65 | 536.04 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-12-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:05:00 | 540.65 | 539.36 | 0.00 | ORB-long ORB[534.10,538.25] vol=1.7x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 539.30 | 539.41 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 11:15:00 | 539.50 | 535.55 | 0.00 | ORB-long ORB[532.10,539.20] vol=2.0x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-12-30 11:55:00 | 537.95 | 536.25 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:00:00 | 522.70 | 525.63 | 0.00 | ORB-short ORB[527.40,530.50] vol=1.8x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-01-01 10:10:00 | 524.22 | 525.47 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:00:00 | 519.00 | 521.16 | 0.00 | ORB-short ORB[525.05,529.20] vol=16.3x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-01-03 11:20:00 | 520.35 | 521.02 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-01-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:10:00 | 500.40 | 502.59 | 0.00 | ORB-short ORB[501.35,505.40] vol=2.5x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 11:10:00 | 497.69 | 502.04 | 0.00 | T1 1.5R @ 497.69 |
| Target hit | 2025-01-07 15:20:00 | 498.00 | 500.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2025-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:40:00 | 449.00 | 449.89 | 0.00 | ORB-short ORB[449.50,454.90] vol=2.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-01-20 09:45:00 | 450.24 | 449.94 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:05:00 | 431.70 | 433.85 | 0.00 | ORB-short ORB[436.65,440.80] vol=4.3x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-01-24 10:40:00 | 432.97 | 433.32 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-01-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:10:00 | 427.85 | 427.07 | 0.00 | ORB-long ORB[422.90,426.80] vol=1.6x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-01-29 10:55:00 | 426.43 | 426.80 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 11:15:00 | 427.25 | 423.42 | 0.00 | ORB-long ORB[421.55,426.70] vol=2.0x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 12:35:00 | 429.11 | 425.17 | 0.00 | T1 1.5R @ 429.11 |
| Target hit | 2025-01-30 15:20:00 | 432.50 | 427.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2025-02-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:00:00 | 433.15 | 436.02 | 0.00 | ORB-short ORB[435.55,441.25] vol=1.5x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:15:00 | 431.07 | 434.83 | 0.00 | T1 1.5R @ 431.07 |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 433.15 | 433.84 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 10:15:00 | 411.30 | 412.77 | 0.00 | ORB-short ORB[414.05,419.65] vol=1.6x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-02-12 10:35:00 | 413.05 | 412.75 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-02-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 11:05:00 | 421.80 | 420.29 | 0.00 | ORB-long ORB[412.40,418.50] vol=1.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-02-13 11:25:00 | 420.47 | 420.36 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-02-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:35:00 | 414.70 | 418.81 | 0.00 | ORB-short ORB[418.10,424.30] vol=2.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:00:00 | 412.39 | 417.63 | 0.00 | T1 1.5R @ 412.39 |
| Stop hit — per-position SL triggered | 2025-02-14 11:55:00 | 414.70 | 416.30 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:55:00 | 414.20 | 412.13 | 0.00 | ORB-long ORB[409.30,413.20] vol=1.6x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:35:00 | 415.84 | 412.89 | 0.00 | T1 1.5R @ 415.84 |
| Stop hit — per-position SL triggered | 2025-02-20 11:50:00 | 414.20 | 413.11 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-02-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:35:00 | 412.75 | 418.49 | 0.00 | ORB-short ORB[417.55,422.70] vol=2.1x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 11:05:00 | 410.58 | 417.58 | 0.00 | T1 1.5R @ 410.58 |
| Target hit | 2025-02-21 15:20:00 | 408.40 | 414.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — SELL (started 2025-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 10:00:00 | 394.75 | 396.26 | 0.00 | ORB-short ORB[395.30,400.20] vol=2.2x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 10:20:00 | 392.90 | 395.85 | 0.00 | T1 1.5R @ 392.90 |
| Target hit | 2025-02-27 15:20:00 | 389.90 | 391.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2025-03-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:30:00 | 386.20 | 383.02 | 0.00 | ORB-long ORB[380.00,383.00] vol=1.9x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 10:40:00 | 388.24 | 383.63 | 0.00 | T1 1.5R @ 388.24 |
| Stop hit — per-position SL triggered | 2025-03-05 10:45:00 | 386.20 | 383.75 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-03-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:40:00 | 405.35 | 403.00 | 0.00 | ORB-long ORB[397.65,403.15] vol=6.8x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 10:05:00 | 408.05 | 403.69 | 0.00 | T1 1.5R @ 408.05 |
| Target hit | 2025-03-07 15:20:00 | 411.35 | 406.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — BUY (started 2025-03-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:40:00 | 408.90 | 405.81 | 0.00 | ORB-long ORB[402.30,406.80] vol=3.0x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-03-12 11:30:00 | 407.30 | 408.08 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 10:40:00 | 397.00 | 400.65 | 0.00 | ORB-short ORB[398.55,404.00] vol=1.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-03-17 10:45:00 | 398.10 | 400.43 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:50:00 | 416.75 | 413.74 | 0.00 | ORB-long ORB[410.00,413.70] vol=2.9x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:55:00 | 418.70 | 415.56 | 0.00 | T1 1.5R @ 418.70 |
| Target hit | 2025-03-19 15:20:00 | 425.05 | 421.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — SELL (started 2025-03-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 09:30:00 | 425.95 | 428.32 | 0.00 | ORB-short ORB[427.00,430.80] vol=1.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 09:35:00 | 424.01 | 427.78 | 0.00 | T1 1.5R @ 424.01 |
| Stop hit — per-position SL triggered | 2025-03-24 11:30:00 | 425.95 | 426.00 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 11:15:00 | 426.15 | 423.09 | 0.00 | ORB-long ORB[417.75,422.40] vol=2.5x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 14:25:00 | 427.92 | 424.18 | 0.00 | T1 1.5R @ 427.92 |
| Target hit | 2025-04-02 15:20:00 | 428.20 | 424.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — BUY (started 2025-04-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:20:00 | 434.55 | 431.53 | 0.00 | ORB-long ORB[428.60,431.00] vol=2.3x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-04-15 10:35:00 | 433.05 | 432.34 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:25:00 | 450.00 | 447.00 | 0.00 | ORB-long ORB[444.45,449.00] vol=3.0x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 12:20:00 | 452.18 | 448.66 | 0.00 | T1 1.5R @ 452.18 |
| Target hit | 2025-04-17 15:20:00 | 451.05 | 450.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 456.75 | 453.55 | 0.00 | ORB-long ORB[448.20,454.60] vol=2.4x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 13:55:00 | 459.46 | 456.16 | 0.00 | T1 1.5R @ 459.46 |
| Target hit | 2025-04-21 15:20:00 | 460.15 | 458.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 90 — BUY (started 2025-04-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:25:00 | 465.05 | 462.57 | 0.00 | ORB-long ORB[457.95,462.75] vol=2.3x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-04-22 10:35:00 | 463.71 | 462.86 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2025-04-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:50:00 | 459.40 | 462.85 | 0.00 | ORB-short ORB[462.15,465.55] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-04-23 12:45:00 | 461.03 | 461.94 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2025-04-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:20:00 | 463.10 | 466.08 | 0.00 | ORB-short ORB[469.15,474.35] vol=2.8x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:25:00 | 460.40 | 464.43 | 0.00 | T1 1.5R @ 460.40 |
| Stop hit — per-position SL triggered | 2025-04-25 12:45:00 | 463.10 | 462.87 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 457.35 | 460.89 | 0.00 | ORB-short ORB[461.05,464.30] vol=3.9x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:35:00 | 455.18 | 459.50 | 0.00 | T1 1.5R @ 455.18 |
| Stop hit — per-position SL triggered | 2025-04-29 09:45:00 | 457.35 | 458.44 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2025-05-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 10:05:00 | 478.00 | 475.81 | 0.00 | ORB-long ORB[472.30,477.00] vol=1.6x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-05-02 10:15:00 | 476.26 | 476.08 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:05:00 | 489.55 | 486.17 | 0.00 | ORB-long ORB[481.65,487.40] vol=1.6x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-05-05 11:20:00 | 487.78 | 486.40 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2025-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:45:00 | 488.60 | 485.64 | 0.00 | ORB-long ORB[482.75,487.80] vol=2.1x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-05-06 10:55:00 | 487.21 | 485.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-23 10:35:00 | 483.30 | 2024-05-23 15:20:00 | 483.50 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest1 | 2024-05-24 11:15:00 | 486.30 | 2024-05-24 11:25:00 | 485.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-27 09:30:00 | 474.75 | 2024-05-27 09:35:00 | 476.47 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-30 09:40:00 | 476.45 | 2024-05-30 09:45:00 | 475.03 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-07 10:10:00 | 476.60 | 2024-06-07 10:15:00 | 478.62 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-06-07 10:10:00 | 476.60 | 2024-06-07 12:50:00 | 480.75 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2024-06-10 10:50:00 | 484.70 | 2024-06-10 11:05:00 | 486.23 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-13 09:45:00 | 481.20 | 2024-06-13 09:50:00 | 479.99 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-14 09:35:00 | 473.30 | 2024-06-14 10:00:00 | 474.36 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-06-18 09:40:00 | 483.85 | 2024-06-18 10:50:00 | 486.04 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-18 09:40:00 | 483.85 | 2024-06-18 15:10:00 | 484.15 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2024-06-27 10:45:00 | 512.35 | 2024-06-27 11:15:00 | 513.84 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-28 09:30:00 | 521.80 | 2024-06-28 09:50:00 | 519.81 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-02 10:15:00 | 540.85 | 2024-07-02 10:40:00 | 542.71 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-03 09:40:00 | 533.15 | 2024-07-03 10:45:00 | 530.52 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-03 09:40:00 | 533.15 | 2024-07-03 10:50:00 | 533.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 09:35:00 | 528.35 | 2024-07-05 09:40:00 | 530.81 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-05 09:35:00 | 528.35 | 2024-07-05 09:50:00 | 528.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 10:10:00 | 528.40 | 2024-07-09 10:15:00 | 526.82 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-16 09:30:00 | 545.35 | 2024-07-16 09:35:00 | 543.68 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-18 09:35:00 | 545.65 | 2024-07-18 09:40:00 | 547.46 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-22 10:55:00 | 528.00 | 2024-07-22 11:00:00 | 525.91 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-24 09:35:00 | 524.95 | 2024-07-24 09:50:00 | 528.25 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-07-24 09:35:00 | 524.95 | 2024-07-24 15:20:00 | 538.95 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2024-07-25 09:50:00 | 538.35 | 2024-07-25 10:15:00 | 536.21 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-30 10:15:00 | 556.40 | 2024-07-30 10:20:00 | 559.72 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-07-30 10:15:00 | 556.40 | 2024-07-30 10:25:00 | 556.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-01 09:30:00 | 552.65 | 2024-08-01 09:45:00 | 554.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-07 10:50:00 | 517.00 | 2024-08-07 11:00:00 | 519.53 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-08-13 10:20:00 | 489.95 | 2024-08-13 10:55:00 | 491.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-16 10:15:00 | 481.70 | 2024-08-16 14:05:00 | 483.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-20 10:35:00 | 489.30 | 2024-08-20 10:45:00 | 488.07 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-23 09:30:00 | 502.05 | 2024-08-23 09:35:00 | 503.27 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-27 10:50:00 | 511.35 | 2024-08-27 11:00:00 | 510.03 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-29 10:25:00 | 499.95 | 2024-08-29 10:35:00 | 499.06 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-09-05 10:10:00 | 514.80 | 2024-09-05 10:15:00 | 513.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-06 09:45:00 | 506.20 | 2024-09-06 10:05:00 | 504.12 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-09-06 09:45:00 | 506.20 | 2024-09-06 13:25:00 | 505.80 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2024-09-16 09:55:00 | 524.65 | 2024-09-16 10:00:00 | 526.56 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-17 09:40:00 | 519.30 | 2024-09-17 09:55:00 | 517.09 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-17 09:40:00 | 519.30 | 2024-09-17 10:00:00 | 519.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 09:40:00 | 517.05 | 2024-09-19 09:50:00 | 515.36 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-24 10:05:00 | 522.15 | 2024-09-24 10:15:00 | 524.25 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-24 10:05:00 | 522.15 | 2024-09-24 15:20:00 | 529.85 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2024-10-07 11:05:00 | 505.85 | 2024-10-07 11:10:00 | 508.02 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-08 09:45:00 | 500.75 | 2024-10-08 09:55:00 | 497.85 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-10-09 11:05:00 | 512.35 | 2024-10-09 11:20:00 | 514.10 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-11 11:15:00 | 502.30 | 2024-10-11 11:50:00 | 503.29 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-10-15 09:35:00 | 512.50 | 2024-10-15 09:40:00 | 514.72 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-10-15 09:35:00 | 512.50 | 2024-10-15 10:00:00 | 513.50 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-10-16 11:05:00 | 510.30 | 2024-10-16 11:25:00 | 511.64 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-21 10:55:00 | 510.50 | 2024-10-21 11:00:00 | 513.07 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-21 10:55:00 | 510.50 | 2024-10-21 11:30:00 | 510.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-23 10:15:00 | 499.75 | 2024-10-23 11:15:00 | 504.03 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2024-10-23 10:15:00 | 499.75 | 2024-10-23 12:15:00 | 499.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 10:35:00 | 480.20 | 2024-10-29 10:55:00 | 482.04 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-30 09:40:00 | 502.40 | 2024-10-30 09:50:00 | 500.37 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-31 09:30:00 | 501.25 | 2024-10-31 09:35:00 | 499.42 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-07 09:40:00 | 496.80 | 2024-11-07 09:50:00 | 495.16 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-19 09:30:00 | 480.55 | 2024-11-19 09:40:00 | 478.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-21 10:50:00 | 481.95 | 2024-11-21 11:10:00 | 484.36 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-11-21 10:50:00 | 481.95 | 2024-11-21 14:35:00 | 481.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:50:00 | 487.40 | 2024-11-22 11:20:00 | 485.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-26 09:40:00 | 510.00 | 2024-11-26 09:55:00 | 508.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-28 10:55:00 | 508.00 | 2024-11-28 11:15:00 | 506.22 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-11-28 10:55:00 | 508.00 | 2024-11-28 11:30:00 | 508.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 10:10:00 | 504.65 | 2024-11-29 10:25:00 | 506.26 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-03 09:40:00 | 517.45 | 2024-12-03 09:50:00 | 516.22 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-04 10:45:00 | 524.50 | 2024-12-04 11:05:00 | 527.24 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-12-04 10:45:00 | 524.50 | 2024-12-04 15:20:00 | 534.90 | TARGET_HIT | 0.50 | 1.98% |
| BUY | retest1 | 2024-12-05 09:45:00 | 542.00 | 2024-12-05 10:55:00 | 540.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-09 09:35:00 | 544.10 | 2024-12-09 09:45:00 | 546.44 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-09 09:35:00 | 544.10 | 2024-12-09 12:25:00 | 550.50 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2024-12-10 09:55:00 | 539.15 | 2024-12-10 10:10:00 | 540.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-12 09:55:00 | 542.85 | 2024-12-12 10:05:00 | 541.52 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-13 10:15:00 | 540.00 | 2024-12-13 10:55:00 | 541.04 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-17 10:40:00 | 536.10 | 2024-12-17 11:15:00 | 534.14 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-17 10:40:00 | 536.10 | 2024-12-17 15:20:00 | 532.55 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-12-20 10:45:00 | 534.00 | 2024-12-20 12:20:00 | 532.51 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-24 11:10:00 | 535.65 | 2024-12-24 11:20:00 | 537.43 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-12-24 11:10:00 | 535.65 | 2024-12-24 12:20:00 | 535.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-26 10:05:00 | 540.65 | 2024-12-26 10:15:00 | 539.30 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-30 11:15:00 | 539.50 | 2024-12-30 11:55:00 | 537.95 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-01 10:00:00 | 522.70 | 2025-01-01 10:10:00 | 524.22 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-03 11:00:00 | 519.00 | 2025-01-03 11:20:00 | 520.35 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-07 10:10:00 | 500.40 | 2025-01-07 11:10:00 | 497.69 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-07 10:10:00 | 500.40 | 2025-01-07 15:20:00 | 498.00 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-20 09:40:00 | 449.00 | 2025-01-20 09:45:00 | 450.24 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-24 10:05:00 | 431.70 | 2025-01-24 10:40:00 | 432.97 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-29 10:10:00 | 427.85 | 2025-01-29 10:55:00 | 426.43 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-30 11:15:00 | 427.25 | 2025-01-30 12:35:00 | 429.11 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-30 11:15:00 | 427.25 | 2025-01-30 15:20:00 | 432.50 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2025-02-01 10:00:00 | 433.15 | 2025-02-01 10:15:00 | 431.07 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-02-01 10:00:00 | 433.15 | 2025-02-01 11:15:00 | 433.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-12 10:15:00 | 411.30 | 2025-02-12 10:35:00 | 413.05 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-02-13 11:05:00 | 421.80 | 2025-02-13 11:25:00 | 420.47 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-14 10:35:00 | 414.70 | 2025-02-14 11:00:00 | 412.39 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-14 10:35:00 | 414.70 | 2025-02-14 11:55:00 | 414.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 10:55:00 | 414.20 | 2025-02-20 11:35:00 | 415.84 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-02-20 10:55:00 | 414.20 | 2025-02-20 11:50:00 | 414.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-21 10:35:00 | 412.75 | 2025-02-21 11:05:00 | 410.58 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-02-21 10:35:00 | 412.75 | 2025-02-21 15:20:00 | 408.40 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2025-02-27 10:00:00 | 394.75 | 2025-02-27 10:20:00 | 392.90 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-02-27 10:00:00 | 394.75 | 2025-02-27 15:20:00 | 389.90 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2025-03-05 10:30:00 | 386.20 | 2025-03-05 10:40:00 | 388.24 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-05 10:30:00 | 386.20 | 2025-03-05 10:45:00 | 386.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 09:40:00 | 405.35 | 2025-03-07 10:05:00 | 408.05 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-03-07 09:40:00 | 405.35 | 2025-03-07 15:20:00 | 411.35 | TARGET_HIT | 0.50 | 1.48% |
| BUY | retest1 | 2025-03-12 09:40:00 | 408.90 | 2025-03-12 11:30:00 | 407.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-03-17 10:40:00 | 397.00 | 2025-03-17 10:45:00 | 398.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-19 09:50:00 | 416.75 | 2025-03-19 10:55:00 | 418.70 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-03-19 09:50:00 | 416.75 | 2025-03-19 15:20:00 | 425.05 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2025-03-24 09:30:00 | 425.95 | 2025-03-24 09:35:00 | 424.01 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-03-24 09:30:00 | 425.95 | 2025-03-24 11:30:00 | 425.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 11:15:00 | 426.15 | 2025-04-02 14:25:00 | 427.92 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-04-02 11:15:00 | 426.15 | 2025-04-02 15:20:00 | 428.20 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-15 10:20:00 | 434.55 | 2025-04-15 10:35:00 | 433.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-17 10:25:00 | 450.00 | 2025-04-17 12:20:00 | 452.18 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-17 10:25:00 | 450.00 | 2025-04-17 15:20:00 | 451.05 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2025-04-21 09:35:00 | 456.75 | 2025-04-21 13:55:00 | 459.46 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-21 09:35:00 | 456.75 | 2025-04-21 15:20:00 | 460.15 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2025-04-22 10:25:00 | 465.05 | 2025-04-22 10:35:00 | 463.71 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-23 10:50:00 | 459.40 | 2025-04-23 12:45:00 | 461.03 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-25 10:20:00 | 463.10 | 2025-04-25 10:25:00 | 460.40 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-04-25 10:20:00 | 463.10 | 2025-04-25 12:45:00 | 463.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-29 09:30:00 | 457.35 | 2025-04-29 09:35:00 | 455.18 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-04-29 09:30:00 | 457.35 | 2025-04-29 09:45:00 | 457.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-02 10:05:00 | 478.00 | 2025-05-02 10:15:00 | 476.26 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-05 11:05:00 | 489.55 | 2025-05-05 11:20:00 | 487.78 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-06 10:45:00 | 488.60 | 2025-05-06 10:55:00 | 487.21 | STOP_HIT | 1.00 | -0.28% |
