# Asahi India Glass Ltd. (ASAHIINDIA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-09-26 15:25:00 (41166 bars)
- **Last close:** 897.00
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
| ENTRY1 | 110 |
| ENTRY2 | 0 |
| PARTIAL | 43 |
| TARGET_HIT | 20 |
| STOP_HIT | 90 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 153 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 90
- **Target hits / Stop hits / Partials:** 20 / 90 / 43
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 28.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 24 | 43.6% | 8 | 31 | 16 | 0.23% | 12.6% |
| BUY @ 2nd Alert (retest1) | 55 | 24 | 43.6% | 8 | 31 | 16 | 0.23% | 12.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 98 | 39 | 39.8% | 12 | 59 | 27 | 0.16% | 15.9% |
| SELL @ 2nd Alert (retest1) | 98 | 39 | 39.8% | 12 | 59 | 27 | 0.16% | 15.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 153 | 63 | 41.2% | 20 | 90 | 43 | 0.19% | 28.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 11:00:00 | 468.70 | 472.57 | 0.00 | ORB-short ORB[471.65,477.70] vol=10.6x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 11:55:00 | 466.29 | 471.06 | 0.00 | T1 1.5R @ 466.29 |
| Stop hit — per-position SL triggered | 2023-05-17 15:15:00 | 468.70 | 469.20 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:35:00 | 454.70 | 457.64 | 0.00 | ORB-short ORB[457.95,463.40] vol=1.9x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-05-19 09:45:00 | 456.35 | 456.95 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 10:05:00 | 461.00 | 458.03 | 0.00 | ORB-long ORB[453.80,458.60] vol=4.0x ATR=1.54 |
| Stop hit — per-position SL triggered | 2023-05-24 13:00:00 | 459.46 | 459.38 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:50:00 | 464.40 | 462.27 | 0.00 | ORB-long ORB[457.35,460.00] vol=8.4x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-05-25 10:00:00 | 462.57 | 462.49 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:30:00 | 457.15 | 459.48 | 0.00 | ORB-short ORB[458.70,464.25] vol=2.5x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 09:55:00 | 455.19 | 458.34 | 0.00 | T1 1.5R @ 455.19 |
| Stop hit — per-position SL triggered | 2023-05-26 10:00:00 | 457.15 | 458.32 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 10:40:00 | 458.05 | 459.64 | 0.00 | ORB-short ORB[458.35,463.65] vol=2.3x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-05-29 11:45:00 | 459.05 | 459.29 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 11:00:00 | 454.75 | 452.61 | 0.00 | ORB-long ORB[448.85,453.40] vol=2.8x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-06-01 11:05:00 | 453.74 | 452.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 10:55:00 | 456.25 | 454.93 | 0.00 | ORB-long ORB[453.15,456.00] vol=4.3x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 11:30:00 | 457.70 | 455.26 | 0.00 | T1 1.5R @ 457.70 |
| Stop hit — per-position SL triggered | 2023-06-05 11:40:00 | 456.25 | 455.28 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 09:30:00 | 472.00 | 473.52 | 0.00 | ORB-short ORB[473.05,475.05] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2023-06-08 09:45:00 | 473.27 | 473.20 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 471.05 | 472.71 | 0.00 | ORB-short ORB[472.70,474.95] vol=1.9x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 09:35:00 | 469.05 | 470.90 | 0.00 | T1 1.5R @ 469.05 |
| Stop hit — per-position SL triggered | 2023-06-09 09:45:00 | 471.05 | 470.62 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:55:00 | 493.00 | 492.50 | 0.00 | ORB-long ORB[488.60,492.35] vol=3.1x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:05:00 | 496.02 | 493.04 | 0.00 | T1 1.5R @ 496.02 |
| Target hit | 2023-06-20 13:40:00 | 494.00 | 494.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2023-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-26 09:55:00 | 488.95 | 492.57 | 0.00 | ORB-short ORB[489.35,494.90] vol=5.2x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 10:00:00 | 484.97 | 491.45 | 0.00 | T1 1.5R @ 484.97 |
| Stop hit — per-position SL triggered | 2023-06-26 10:10:00 | 488.95 | 490.91 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 10:35:00 | 488.10 | 486.21 | 0.00 | ORB-long ORB[483.00,487.00] vol=2.8x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 12:30:00 | 490.14 | 488.22 | 0.00 | T1 1.5R @ 490.14 |
| Stop hit — per-position SL triggered | 2023-06-28 12:45:00 | 488.10 | 488.32 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 10:05:00 | 489.25 | 486.62 | 0.00 | ORB-long ORB[483.05,485.95] vol=2.9x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-06-30 11:20:00 | 487.45 | 487.36 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 09:30:00 | 496.00 | 493.59 | 0.00 | ORB-long ORB[489.00,493.85] vol=4.1x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 09:45:00 | 498.68 | 496.38 | 0.00 | T1 1.5R @ 498.68 |
| Target hit | 2023-07-03 11:15:00 | 509.80 | 510.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2023-07-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 09:40:00 | 508.60 | 513.23 | 0.00 | ORB-short ORB[512.05,517.15] vol=2.1x ATR=2.33 |
| Stop hit — per-position SL triggered | 2023-07-05 09:45:00 | 510.93 | 512.57 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 10:50:00 | 533.00 | 537.00 | 0.00 | ORB-short ORB[536.05,538.95] vol=2.1x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 11:45:00 | 530.11 | 536.35 | 0.00 | T1 1.5R @ 530.11 |
| Stop hit — per-position SL triggered | 2023-07-06 13:40:00 | 533.00 | 535.09 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:40:00 | 536.30 | 534.87 | 0.00 | ORB-long ORB[528.25,535.30] vol=5.8x ATR=1.98 |
| Stop hit — per-position SL triggered | 2023-07-07 09:55:00 | 534.32 | 535.56 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 09:40:00 | 539.10 | 533.79 | 0.00 | ORB-long ORB[528.05,532.45] vol=2.2x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-07-10 09:45:00 | 536.35 | 534.18 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 10:20:00 | 528.20 | 529.24 | 0.00 | ORB-short ORB[528.50,533.95] vol=2.2x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-07-12 10:35:00 | 530.09 | 529.49 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:50:00 | 530.15 | 534.83 | 0.00 | ORB-short ORB[533.30,539.70] vol=2.5x ATR=1.96 |
| Stop hit — per-position SL triggered | 2023-07-13 11:20:00 | 532.11 | 534.51 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:35:00 | 531.55 | 533.44 | 0.00 | ORB-short ORB[533.10,536.00] vol=2.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-07-18 10:30:00 | 533.39 | 532.04 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 09:35:00 | 526.45 | 527.38 | 0.00 | ORB-short ORB[527.05,531.40] vol=6.5x ATR=1.60 |
| Stop hit — per-position SL triggered | 2023-07-19 14:05:00 | 528.05 | 526.42 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:50:00 | 531.00 | 532.81 | 0.00 | ORB-short ORB[532.90,536.55] vol=1.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-07-20 09:55:00 | 533.09 | 532.75 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:45:00 | 565.25 | 561.33 | 0.00 | ORB-long ORB[555.05,563.50] vol=2.8x ATR=2.56 |
| Stop hit — per-position SL triggered | 2023-07-24 09:55:00 | 562.69 | 561.93 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:50:00 | 562.45 | 560.53 | 0.00 | ORB-long ORB[555.00,561.60] vol=1.7x ATR=1.86 |
| Stop hit — per-position SL triggered | 2023-07-26 11:05:00 | 560.59 | 560.70 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-07-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 11:00:00 | 555.70 | 558.19 | 0.00 | ORB-short ORB[558.55,564.85] vol=3.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-07-27 11:10:00 | 557.37 | 558.05 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 10:35:00 | 535.50 | 539.16 | 0.00 | ORB-short ORB[536.50,543.00] vol=2.7x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 10:40:00 | 531.78 | 538.51 | 0.00 | T1 1.5R @ 531.78 |
| Target hit | 2023-08-03 15:20:00 | 529.00 | 532.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2023-08-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:40:00 | 533.15 | 536.81 | 0.00 | ORB-short ORB[534.00,540.70] vol=2.5x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 12:40:00 | 530.88 | 534.64 | 0.00 | T1 1.5R @ 530.88 |
| Stop hit — per-position SL triggered | 2023-08-07 15:10:00 | 533.15 | 532.99 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:30:00 | 534.25 | 532.96 | 0.00 | ORB-long ORB[530.05,534.00] vol=3.2x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-08-10 10:20:00 | 532.43 | 533.33 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:35:00 | 510.00 | 507.84 | 0.00 | ORB-long ORB[504.55,509.20] vol=1.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-08-16 09:55:00 | 508.17 | 507.73 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-21 10:05:00 | 496.00 | 497.34 | 0.00 | ORB-short ORB[496.35,501.55] vol=3.0x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-21 10:10:00 | 493.73 | 496.83 | 0.00 | T1 1.5R @ 493.73 |
| Target hit | 2023-08-21 15:20:00 | 492.85 | 494.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2023-08-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:55:00 | 497.25 | 495.00 | 0.00 | ORB-long ORB[493.05,496.05] vol=1.5x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 10:00:00 | 499.61 | 495.83 | 0.00 | T1 1.5R @ 499.61 |
| Target hit | 2023-08-22 15:20:00 | 507.00 | 502.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2023-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 09:30:00 | 518.05 | 515.41 | 0.00 | ORB-long ORB[509.85,516.50] vol=3.5x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 09:50:00 | 520.71 | 518.21 | 0.00 | T1 1.5R @ 520.71 |
| Stop hit — per-position SL triggered | 2023-08-25 10:30:00 | 518.05 | 520.15 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 10:15:00 | 590.55 | 586.13 | 0.00 | ORB-long ORB[584.00,590.05] vol=1.9x ATR=2.72 |
| Stop hit — per-position SL triggered | 2023-08-31 14:20:00 | 587.83 | 588.72 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 10:25:00 | 572.35 | 574.71 | 0.00 | ORB-short ORB[573.85,580.00] vol=1.7x ATR=1.24 |
| Stop hit — per-position SL triggered | 2023-09-05 10:40:00 | 573.59 | 574.62 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 11:00:00 | 578.15 | 574.63 | 0.00 | ORB-long ORB[570.50,578.05] vol=1.7x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 11:05:00 | 580.89 | 576.04 | 0.00 | T1 1.5R @ 580.89 |
| Target hit | 2023-09-15 15:20:00 | 605.70 | 592.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2023-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 09:30:00 | 612.00 | 615.00 | 0.00 | ORB-short ORB[613.50,620.95] vol=2.0x ATR=2.49 |
| Stop hit — per-position SL triggered | 2023-09-26 09:45:00 | 614.49 | 614.14 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:35:00 | 620.40 | 617.54 | 0.00 | ORB-long ORB[612.00,617.45] vol=2.0x ATR=2.47 |
| Stop hit — per-position SL triggered | 2023-09-27 09:40:00 | 617.93 | 617.65 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 11:10:00 | 609.65 | 612.59 | 0.00 | ORB-short ORB[614.30,617.95] vol=2.0x ATR=1.61 |
| Stop hit — per-position SL triggered | 2023-10-05 12:00:00 | 611.26 | 612.32 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 11:15:00 | 602.75 | 605.90 | 0.00 | ORB-short ORB[604.00,610.00] vol=2.4x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-10-11 11:20:00 | 603.91 | 605.89 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 10:20:00 | 594.00 | 595.91 | 0.00 | ORB-short ORB[595.10,604.00] vol=1.8x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 13:35:00 | 591.24 | 593.82 | 0.00 | T1 1.5R @ 591.24 |
| Target hit | 2023-10-16 15:20:00 | 588.50 | 592.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 10:15:00 | 589.95 | 591.90 | 0.00 | ORB-short ORB[590.20,594.90] vol=2.0x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 10:30:00 | 587.94 | 591.63 | 0.00 | T1 1.5R @ 587.94 |
| Stop hit — per-position SL triggered | 2023-10-17 11:55:00 | 589.95 | 591.11 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:50:00 | 586.05 | 591.75 | 0.00 | ORB-short ORB[589.30,594.65] vol=3.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 587.72 | 591.03 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 09:40:00 | 593.50 | 589.48 | 0.00 | ORB-long ORB[582.25,588.35] vol=4.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2023-10-20 09:55:00 | 591.37 | 590.52 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:50:00 | 570.00 | 561.83 | 0.00 | ORB-long ORB[554.50,562.35] vol=5.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2023-11-02 10:55:00 | 567.99 | 563.33 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 11:15:00 | 553.90 | 557.87 | 0.00 | ORB-short ORB[555.00,559.80] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2023-11-03 13:10:00 | 555.87 | 557.04 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 11:15:00 | 552.75 | 555.44 | 0.00 | ORB-short ORB[553.90,557.00] vol=4.8x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 11:25:00 | 550.96 | 555.16 | 0.00 | T1 1.5R @ 550.96 |
| Stop hit — per-position SL triggered | 2023-11-06 11:50:00 | 552.75 | 554.50 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:10:00 | 557.35 | 553.74 | 0.00 | ORB-long ORB[549.35,552.85] vol=4.3x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 10:15:00 | 559.93 | 555.40 | 0.00 | T1 1.5R @ 559.93 |
| Target hit | 2023-11-08 13:45:00 | 559.75 | 561.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2023-11-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:25:00 | 556.90 | 561.06 | 0.00 | ORB-short ORB[561.20,566.70] vol=4.0x ATR=1.49 |
| Stop hit — per-position SL triggered | 2023-11-09 11:05:00 | 558.39 | 560.11 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 11:15:00 | 583.30 | 578.61 | 0.00 | ORB-long ORB[575.15,583.00] vol=3.0x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 11:45:00 | 586.65 | 579.38 | 0.00 | T1 1.5R @ 586.65 |
| Stop hit — per-position SL triggered | 2023-11-13 11:50:00 | 583.30 | 579.43 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-15 09:45:00 | 581.10 | 583.11 | 0.00 | ORB-short ORB[581.15,586.15] vol=1.6x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 10:25:00 | 578.58 | 582.34 | 0.00 | T1 1.5R @ 578.58 |
| Target hit | 2023-11-15 15:20:00 | 573.05 | 575.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2023-11-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-16 09:30:00 | 573.35 | 575.04 | 0.00 | ORB-short ORB[574.30,577.00] vol=2.3x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 10:00:00 | 570.75 | 573.76 | 0.00 | T1 1.5R @ 570.75 |
| Stop hit — per-position SL triggered | 2023-11-16 11:05:00 | 573.35 | 572.88 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 09:30:00 | 574.05 | 573.45 | 0.00 | ORB-long ORB[568.25,572.75] vol=4.8x ATR=2.14 |
| Stop hit — per-position SL triggered | 2023-11-17 10:05:00 | 571.91 | 573.21 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:45:00 | 567.25 | 570.32 | 0.00 | ORB-short ORB[568.00,572.65] vol=1.9x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-11-20 09:50:00 | 569.05 | 570.30 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 10:10:00 | 562.90 | 565.64 | 0.00 | ORB-short ORB[565.15,569.75] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-11-21 10:50:00 | 564.85 | 565.26 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 10:35:00 | 575.75 | 573.02 | 0.00 | ORB-long ORB[570.00,574.40] vol=4.7x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 10:45:00 | 578.13 | 573.75 | 0.00 | T1 1.5R @ 578.13 |
| Stop hit — per-position SL triggered | 2023-11-22 11:45:00 | 575.75 | 576.03 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-11-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 11:05:00 | 575.55 | 576.50 | 0.00 | ORB-short ORB[575.75,579.95] vol=2.8x ATR=1.13 |
| Stop hit — per-position SL triggered | 2023-11-23 15:10:00 | 576.68 | 576.21 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 580.00 | 578.85 | 0.00 | ORB-long ORB[575.30,579.10] vol=6.9x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 09:40:00 | 582.44 | 580.25 | 0.00 | T1 1.5R @ 582.44 |
| Target hit | 2023-11-24 09:55:00 | 580.55 | 580.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — SELL (started 2023-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 10:30:00 | 564.70 | 566.81 | 0.00 | ORB-short ORB[565.60,569.35] vol=1.7x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:05:00 | 562.39 | 566.17 | 0.00 | T1 1.5R @ 562.39 |
| Target hit | 2023-11-28 15:20:00 | 558.95 | 561.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2023-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:35:00 | 575.95 | 577.34 | 0.00 | ORB-short ORB[577.60,583.95] vol=1.8x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 09:40:00 | 571.61 | 574.43 | 0.00 | T1 1.5R @ 571.61 |
| Target hit | 2023-12-06 15:20:00 | 560.00 | 565.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 559.40 | 561.96 | 0.00 | ORB-short ORB[561.75,567.50] vol=4.1x ATR=1.38 |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 560.78 | 561.71 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2023-12-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 10:30:00 | 555.35 | 557.78 | 0.00 | ORB-short ORB[556.55,562.00] vol=1.5x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-12-11 11:05:00 | 557.08 | 557.46 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:55:00 | 546.05 | 547.08 | 0.00 | ORB-short ORB[546.20,551.00] vol=3.4x ATR=1.63 |
| Stop hit — per-position SL triggered | 2023-12-13 11:20:00 | 547.68 | 546.82 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:40:00 | 563.55 | 568.74 | 0.00 | ORB-short ORB[568.35,573.70] vol=2.7x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 10:45:00 | 560.53 | 565.58 | 0.00 | T1 1.5R @ 560.53 |
| Stop hit — per-position SL triggered | 2023-12-19 11:00:00 | 563.55 | 565.45 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:30:00 | 569.70 | 569.18 | 0.00 | ORB-long ORB[564.30,569.50] vol=4.5x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-12-20 10:55:00 | 567.49 | 569.15 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:45:00 | 561.10 | 563.67 | 0.00 | ORB-short ORB[562.15,566.00] vol=1.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2023-12-22 11:40:00 | 562.51 | 562.82 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2023-12-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:20:00 | 560.90 | 563.05 | 0.00 | ORB-short ORB[562.35,566.60] vol=1.9x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 11:05:00 | 558.70 | 562.09 | 0.00 | T1 1.5R @ 558.70 |
| Stop hit — per-position SL triggered | 2023-12-26 11:10:00 | 560.90 | 562.03 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:40:00 | 577.25 | 570.95 | 0.00 | ORB-long ORB[561.05,567.45] vol=10.6x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:45:00 | 580.81 | 573.01 | 0.00 | T1 1.5R @ 580.81 |
| Stop hit — per-position SL triggered | 2023-12-28 09:50:00 | 577.25 | 573.32 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 11:10:00 | 568.40 | 568.94 | 0.00 | ORB-short ORB[570.20,575.00] vol=3.0x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 11:50:00 | 565.74 | 568.61 | 0.00 | T1 1.5R @ 565.74 |
| Stop hit — per-position SL triggered | 2024-01-02 12:45:00 | 568.40 | 568.80 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 09:30:00 | 576.45 | 578.16 | 0.00 | ORB-short ORB[577.35,580.00] vol=2.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-01-05 09:35:00 | 577.91 | 578.00 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:40:00 | 564.90 | 567.73 | 0.00 | ORB-short ORB[568.00,572.45] vol=2.0x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 14:55:00 | 562.62 | 565.04 | 0.00 | T1 1.5R @ 562.62 |
| Target hit | 2024-01-08 15:20:00 | 560.35 | 564.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2024-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:30:00 | 561.00 | 562.53 | 0.00 | ORB-short ORB[562.00,566.10] vol=3.0x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 09:45:00 | 558.40 | 561.50 | 0.00 | T1 1.5R @ 558.40 |
| Target hit | 2024-01-09 15:20:00 | 551.60 | 554.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 11:15:00 | 557.40 | 560.86 | 0.00 | ORB-short ORB[557.95,564.95] vol=2.7x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-01-11 11:25:00 | 558.63 | 560.81 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-01-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:30:00 | 569.15 | 566.27 | 0.00 | ORB-long ORB[564.00,568.00] vol=1.8x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-01-12 09:35:00 | 567.42 | 566.41 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:55:00 | 565.00 | 570.42 | 0.00 | ORB-short ORB[571.05,574.90] vol=1.5x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-01-15 10:05:00 | 566.44 | 569.18 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-16 10:05:00 | 561.05 | 562.21 | 0.00 | ORB-short ORB[561.45,565.75] vol=1.5x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-01-16 10:55:00 | 562.39 | 561.92 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 549.50 | 552.18 | 0.00 | ORB-short ORB[552.05,555.40] vol=2.1x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:45:00 | 547.27 | 549.35 | 0.00 | T1 1.5R @ 547.27 |
| Target hit | 2024-01-18 10:05:00 | 546.00 | 545.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 79 — SELL (started 2024-01-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:00:00 | 554.60 | 556.62 | 0.00 | ORB-short ORB[555.40,558.85] vol=2.4x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-01-19 11:25:00 | 555.80 | 556.45 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:50:00 | 546.00 | 549.03 | 0.00 | ORB-short ORB[548.45,553.60] vol=3.0x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-01-20 11:05:00 | 547.49 | 548.55 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-01-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 10:50:00 | 534.75 | 539.96 | 0.00 | ORB-short ORB[540.00,546.60] vol=4.9x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 11:05:00 | 531.77 | 537.64 | 0.00 | T1 1.5R @ 531.77 |
| Target hit | 2024-01-30 15:20:00 | 523.15 | 527.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — BUY (started 2024-02-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:50:00 | 531.50 | 524.28 | 0.00 | ORB-long ORB[520.00,524.10] vol=1.7x ATR=2.89 |
| Stop hit — per-position SL triggered | 2024-02-02 09:55:00 | 528.61 | 525.49 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 10:15:00 | 551.60 | 548.81 | 0.00 | ORB-long ORB[545.20,550.00] vol=1.8x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-02-07 11:30:00 | 549.48 | 550.49 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-02-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 10:35:00 | 558.60 | 555.08 | 0.00 | ORB-long ORB[548.00,554.05] vol=10.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-02-08 10:40:00 | 556.82 | 555.33 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 09:40:00 | 542.15 | 547.24 | 0.00 | ORB-short ORB[547.35,553.25] vol=3.2x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-02-12 09:50:00 | 544.59 | 546.78 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:00:00 | 542.15 | 542.28 | 0.00 | ORB-short ORB[542.75,547.10] vol=2.1x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-02-15 10:25:00 | 543.84 | 541.91 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 09:55:00 | 540.10 | 543.10 | 0.00 | ORB-short ORB[543.75,547.45] vol=3.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-02-19 10:00:00 | 541.73 | 543.00 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:45:00 | 536.80 | 538.45 | 0.00 | ORB-short ORB[538.20,541.90] vol=1.7x ATR=1.37 |
| Stop hit — per-position SL triggered | 2024-02-20 11:00:00 | 538.17 | 538.31 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-02-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 09:30:00 | 537.00 | 538.36 | 0.00 | ORB-short ORB[537.20,539.95] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-02-21 10:00:00 | 538.26 | 538.14 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:45:00 | 531.25 | 533.25 | 0.00 | ORB-short ORB[532.15,538.70] vol=1.8x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-02-26 09:50:00 | 532.53 | 533.33 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-02-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 09:30:00 | 529.45 | 530.68 | 0.00 | ORB-short ORB[529.90,534.50] vol=3.4x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 10:05:00 | 526.71 | 529.81 | 0.00 | T1 1.5R @ 526.71 |
| Stop hit — per-position SL triggered | 2024-02-29 10:55:00 | 529.45 | 529.19 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 10:45:00 | 532.60 | 536.57 | 0.00 | ORB-short ORB[537.10,542.00] vol=2.4x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-03-01 11:40:00 | 534.05 | 534.79 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 09:50:00 | 529.05 | 531.00 | 0.00 | ORB-short ORB[530.55,534.45] vol=2.7x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-03-05 10:35:00 | 530.94 | 530.32 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:10:00 | 528.30 | 531.60 | 0.00 | ORB-short ORB[531.75,535.45] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-03-06 10:20:00 | 529.91 | 531.37 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 09:30:00 | 517.25 | 518.73 | 0.00 | ORB-short ORB[519.30,524.10] vol=2.3x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:55:00 | 513.76 | 516.05 | 0.00 | T1 1.5R @ 513.76 |
| Stop hit — per-position SL triggered | 2024-03-11 12:05:00 | 517.25 | 515.78 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-03-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:25:00 | 518.50 | 519.67 | 0.00 | ORB-short ORB[518.60,524.65] vol=1.7x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-03-12 10:40:00 | 520.06 | 519.40 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:40:00 | 517.35 | 519.40 | 0.00 | ORB-short ORB[517.40,521.95] vol=1.8x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:50:00 | 514.71 | 518.90 | 0.00 | T1 1.5R @ 514.71 |
| Target hit | 2024-03-13 11:35:00 | 514.25 | 513.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 98 — SELL (started 2024-03-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 09:35:00 | 515.30 | 518.33 | 0.00 | ORB-short ORB[519.40,523.15] vol=4.1x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-03-15 09:45:00 | 517.43 | 516.75 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:30:00 | 504.05 | 506.41 | 0.00 | ORB-short ORB[505.30,509.00] vol=3.3x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-03-19 09:35:00 | 505.97 | 506.35 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2024-03-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 09:55:00 | 520.10 | 516.48 | 0.00 | ORB-long ORB[514.80,519.80] vol=2.8x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-03-20 10:00:00 | 517.90 | 516.71 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-03-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 10:10:00 | 524.85 | 522.37 | 0.00 | ORB-long ORB[515.50,523.05] vol=1.6x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 10:20:00 | 527.46 | 523.08 | 0.00 | T1 1.5R @ 527.46 |
| Stop hit — per-position SL triggered | 2024-03-22 10:30:00 | 524.85 | 523.66 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-04-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 09:40:00 | 544.30 | 541.68 | 0.00 | ORB-long ORB[538.05,543.00] vol=2.1x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 10:35:00 | 547.01 | 543.91 | 0.00 | T1 1.5R @ 547.01 |
| Stop hit — per-position SL triggered | 2024-04-01 10:55:00 | 544.30 | 544.29 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-16 09:55:00 | 582.20 | 585.17 | 0.00 | ORB-short ORB[584.00,589.70] vol=3.1x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-04-16 10:00:00 | 584.01 | 584.82 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-04-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:45:00 | 585.30 | 587.39 | 0.00 | ORB-short ORB[586.00,590.90] vol=1.6x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 13:25:00 | 582.14 | 585.38 | 0.00 | T1 1.5R @ 582.14 |
| Target hit | 2024-04-18 15:20:00 | 574.60 | 581.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 105 — BUY (started 2024-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:55:00 | 581.95 | 577.84 | 0.00 | ORB-long ORB[574.90,581.70] vol=2.1x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 11:25:00 | 585.25 | 582.82 | 0.00 | T1 1.5R @ 585.25 |
| Target hit | 2024-04-23 12:45:00 | 584.30 | 584.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 106 — BUY (started 2024-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:30:00 | 615.75 | 612.87 | 0.00 | ORB-long ORB[608.50,614.45] vol=3.3x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-04-24 09:45:00 | 611.77 | 613.24 | 0.00 | SL hit |

### Cycle 107 — SELL (started 2024-04-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:50:00 | 599.00 | 601.65 | 0.00 | ORB-short ORB[601.15,606.40] vol=3.1x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 11:55:00 | 595.77 | 600.50 | 0.00 | T1 1.5R @ 595.77 |
| Stop hit — per-position SL triggered | 2024-04-25 14:45:00 | 599.00 | 598.94 | 0.00 | SL hit |

### Cycle 108 — BUY (started 2024-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:00:00 | 615.00 | 612.23 | 0.00 | ORB-long ORB[609.25,614.80] vol=3.8x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-04-29 10:15:00 | 612.47 | 612.60 | 0.00 | SL hit |

### Cycle 109 — SELL (started 2024-05-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 11:00:00 | 599.40 | 603.99 | 0.00 | ORB-short ORB[603.95,611.30] vol=2.4x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-05-03 12:40:00 | 601.10 | 603.36 | 0.00 | SL hit |

### Cycle 110 — BUY (started 2024-05-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 10:25:00 | 623.10 | 612.99 | 0.00 | ORB-long ORB[600.10,609.00] vol=13.2x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 10:30:00 | 628.66 | 628.55 | 0.00 | T1 1.5R @ 628.66 |
| Target hit | 2024-05-06 11:00:00 | 639.95 | 645.70 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 11:00:00 | 468.70 | 2023-05-17 11:55:00 | 466.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-05-17 11:00:00 | 468.70 | 2023-05-17 15:15:00 | 468.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-19 09:35:00 | 454.70 | 2023-05-19 09:45:00 | 456.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-05-24 10:05:00 | 461.00 | 2023-05-24 13:00:00 | 459.46 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-05-25 09:50:00 | 464.40 | 2023-05-25 10:00:00 | 462.57 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-05-26 09:30:00 | 457.15 | 2023-05-26 09:55:00 | 455.19 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-05-26 09:30:00 | 457.15 | 2023-05-26 10:00:00 | 457.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-29 10:40:00 | 458.05 | 2023-05-29 11:45:00 | 459.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-01 11:00:00 | 454.75 | 2023-06-01 11:05:00 | 453.74 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-05 10:55:00 | 456.25 | 2023-06-05 11:30:00 | 457.70 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-06-05 10:55:00 | 456.25 | 2023-06-05 11:40:00 | 456.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-08 09:30:00 | 472.00 | 2023-06-08 09:45:00 | 473.27 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-09 09:30:00 | 471.05 | 2023-06-09 09:35:00 | 469.05 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-06-09 09:30:00 | 471.05 | 2023-06-09 09:45:00 | 471.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-20 09:55:00 | 493.00 | 2023-06-20 10:05:00 | 496.02 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2023-06-20 09:55:00 | 493.00 | 2023-06-20 13:40:00 | 494.00 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2023-06-26 09:55:00 | 488.95 | 2023-06-26 10:00:00 | 484.97 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2023-06-26 09:55:00 | 488.95 | 2023-06-26 10:10:00 | 488.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-28 10:35:00 | 488.10 | 2023-06-28 12:30:00 | 490.14 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-06-28 10:35:00 | 488.10 | 2023-06-28 12:45:00 | 488.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-30 10:05:00 | 489.25 | 2023-06-30 11:20:00 | 487.45 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-07-03 09:30:00 | 496.00 | 2023-07-03 09:45:00 | 498.68 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-07-03 09:30:00 | 496.00 | 2023-07-03 11:15:00 | 509.80 | TARGET_HIT | 0.50 | 2.78% |
| SELL | retest1 | 2023-07-05 09:40:00 | 508.60 | 2023-07-05 09:45:00 | 510.93 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-07-06 10:50:00 | 533.00 | 2023-07-06 11:45:00 | 530.11 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-07-06 10:50:00 | 533.00 | 2023-07-06 13:40:00 | 533.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-07 09:40:00 | 536.30 | 2023-07-07 09:55:00 | 534.32 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-07-10 09:40:00 | 539.10 | 2023-07-10 09:45:00 | 536.35 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2023-07-12 10:20:00 | 528.20 | 2023-07-12 10:35:00 | 530.09 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-07-13 10:50:00 | 530.15 | 2023-07-13 11:20:00 | 532.11 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-07-18 09:35:00 | 531.55 | 2023-07-18 10:30:00 | 533.39 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-07-19 09:35:00 | 526.45 | 2023-07-19 14:05:00 | 528.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-07-20 09:50:00 | 531.00 | 2023-07-20 09:55:00 | 533.09 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-07-24 09:45:00 | 565.25 | 2023-07-24 09:55:00 | 562.69 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-07-26 10:50:00 | 562.45 | 2023-07-26 11:05:00 | 560.59 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-07-27 11:00:00 | 555.70 | 2023-07-27 11:10:00 | 557.37 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-08-03 10:35:00 | 535.50 | 2023-08-03 10:40:00 | 531.78 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2023-08-03 10:35:00 | 535.50 | 2023-08-03 15:20:00 | 529.00 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2023-08-07 10:40:00 | 533.15 | 2023-08-07 12:40:00 | 530.88 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-08-07 10:40:00 | 533.15 | 2023-08-07 15:10:00 | 533.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-10 09:30:00 | 534.25 | 2023-08-10 10:20:00 | 532.43 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-16 09:35:00 | 510.00 | 2023-08-16 09:55:00 | 508.17 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-08-21 10:05:00 | 496.00 | 2023-08-21 10:10:00 | 493.73 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-08-21 10:05:00 | 496.00 | 2023-08-21 15:20:00 | 492.85 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2023-08-22 09:55:00 | 497.25 | 2023-08-22 10:00:00 | 499.61 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-08-22 09:55:00 | 497.25 | 2023-08-22 15:20:00 | 507.00 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2023-08-25 09:30:00 | 518.05 | 2023-08-25 09:50:00 | 520.71 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-08-25 09:30:00 | 518.05 | 2023-08-25 10:30:00 | 518.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-31 10:15:00 | 590.55 | 2023-08-31 14:20:00 | 587.83 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-09-05 10:25:00 | 572.35 | 2023-09-05 10:40:00 | 573.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-09-15 11:00:00 | 578.15 | 2023-09-15 11:05:00 | 580.89 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-09-15 11:00:00 | 578.15 | 2023-09-15 15:20:00 | 605.70 | TARGET_HIT | 0.50 | 4.77% |
| SELL | retest1 | 2023-09-26 09:30:00 | 612.00 | 2023-09-26 09:45:00 | 614.49 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-09-27 09:35:00 | 620.40 | 2023-09-27 09:40:00 | 617.93 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-10-05 11:10:00 | 609.65 | 2023-10-05 12:00:00 | 611.26 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-11 11:15:00 | 602.75 | 2023-10-11 11:20:00 | 603.91 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-10-16 10:20:00 | 594.00 | 2023-10-16 13:35:00 | 591.24 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-10-16 10:20:00 | 594.00 | 2023-10-16 15:20:00 | 588.50 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2023-10-17 10:15:00 | 589.95 | 2023-10-17 10:30:00 | 587.94 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-10-17 10:15:00 | 589.95 | 2023-10-17 11:55:00 | 589.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-18 10:50:00 | 586.05 | 2023-10-18 11:15:00 | 587.72 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-10-20 09:40:00 | 593.50 | 2023-10-20 09:55:00 | 591.37 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-11-02 10:50:00 | 570.00 | 2023-11-02 10:55:00 | 567.99 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-11-03 11:15:00 | 553.90 | 2023-11-03 13:10:00 | 555.87 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-11-06 11:15:00 | 552.75 | 2023-11-06 11:25:00 | 550.96 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-11-06 11:15:00 | 552.75 | 2023-11-06 11:50:00 | 552.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-08 10:10:00 | 557.35 | 2023-11-08 10:15:00 | 559.93 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-11-08 10:10:00 | 557.35 | 2023-11-08 13:45:00 | 559.75 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-09 10:25:00 | 556.90 | 2023-11-09 11:05:00 | 558.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-11-13 11:15:00 | 583.30 | 2023-11-13 11:45:00 | 586.65 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2023-11-13 11:15:00 | 583.30 | 2023-11-13 11:50:00 | 583.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-15 09:45:00 | 581.10 | 2023-11-15 10:25:00 | 578.58 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-15 09:45:00 | 581.10 | 2023-11-15 15:20:00 | 573.05 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2023-11-16 09:30:00 | 573.35 | 2023-11-16 10:00:00 | 570.75 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-11-16 09:30:00 | 573.35 | 2023-11-16 11:05:00 | 573.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 09:30:00 | 574.05 | 2023-11-17 10:05:00 | 571.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-11-20 09:45:00 | 567.25 | 2023-11-20 09:50:00 | 569.05 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-11-21 10:10:00 | 562.90 | 2023-11-21 10:50:00 | 564.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-22 10:35:00 | 575.75 | 2023-11-22 10:45:00 | 578.13 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-11-22 10:35:00 | 575.75 | 2023-11-22 11:45:00 | 575.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 11:05:00 | 575.55 | 2023-11-23 15:10:00 | 576.68 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-24 09:30:00 | 580.00 | 2023-11-24 09:40:00 | 582.44 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-11-24 09:30:00 | 580.00 | 2023-11-24 09:55:00 | 580.55 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2023-11-28 10:30:00 | 564.70 | 2023-11-28 11:05:00 | 562.39 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-11-28 10:30:00 | 564.70 | 2023-11-28 15:20:00 | 558.95 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2023-12-06 09:35:00 | 575.95 | 2023-12-06 09:40:00 | 571.61 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2023-12-06 09:35:00 | 575.95 | 2023-12-06 15:20:00 | 560.00 | TARGET_HIT | 0.50 | 2.77% |
| SELL | retest1 | 2023-12-08 11:00:00 | 559.40 | 2023-12-08 11:15:00 | 560.78 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-11 10:30:00 | 555.35 | 2023-12-11 11:05:00 | 557.08 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-12-13 09:55:00 | 546.05 | 2023-12-13 11:20:00 | 547.68 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-12-19 10:40:00 | 563.55 | 2023-12-19 10:45:00 | 560.53 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-12-19 10:40:00 | 563.55 | 2023-12-19 11:00:00 | 563.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-20 09:30:00 | 569.70 | 2023-12-20 10:55:00 | 567.49 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-12-22 10:45:00 | 561.10 | 2023-12-22 11:40:00 | 562.51 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-26 10:20:00 | 560.90 | 2023-12-26 11:05:00 | 558.70 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-26 10:20:00 | 560.90 | 2023-12-26 11:10:00 | 560.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-28 09:40:00 | 577.25 | 2023-12-28 09:45:00 | 580.81 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2023-12-28 09:40:00 | 577.25 | 2023-12-28 09:50:00 | 577.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 11:10:00 | 568.40 | 2024-01-02 11:50:00 | 565.74 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-01-02 11:10:00 | 568.40 | 2024-01-02 12:45:00 | 568.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-05 09:30:00 | 576.45 | 2024-01-05 09:35:00 | 577.91 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-08 10:40:00 | 564.90 | 2024-01-08 14:55:00 | 562.62 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-01-08 10:40:00 | 564.90 | 2024-01-08 15:20:00 | 560.35 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2024-01-09 09:30:00 | 561.00 | 2024-01-09 09:45:00 | 558.40 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-01-09 09:30:00 | 561.00 | 2024-01-09 15:20:00 | 551.60 | TARGET_HIT | 0.50 | 1.68% |
| SELL | retest1 | 2024-01-11 11:15:00 | 557.40 | 2024-01-11 11:25:00 | 558.63 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-01-12 09:30:00 | 569.15 | 2024-01-12 09:35:00 | 567.42 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-01-15 09:55:00 | 565.00 | 2024-01-15 10:05:00 | 566.44 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-16 10:05:00 | 561.05 | 2024-01-16 10:55:00 | 562.39 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-18 09:35:00 | 549.50 | 2024-01-18 09:45:00 | 547.27 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-01-18 09:35:00 | 549.50 | 2024-01-18 10:05:00 | 546.00 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2024-01-19 11:00:00 | 554.60 | 2024-01-19 11:25:00 | 555.80 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-20 10:50:00 | 546.00 | 2024-01-20 11:05:00 | 547.49 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-30 10:50:00 | 534.75 | 2024-01-30 11:05:00 | 531.77 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-01-30 10:50:00 | 534.75 | 2024-01-30 15:20:00 | 523.15 | TARGET_HIT | 0.50 | 2.17% |
| BUY | retest1 | 2024-02-02 09:50:00 | 531.50 | 2024-02-02 09:55:00 | 528.61 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-02-07 10:15:00 | 551.60 | 2024-02-07 11:30:00 | 549.48 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-02-08 10:35:00 | 558.60 | 2024-02-08 10:40:00 | 556.82 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-12 09:40:00 | 542.15 | 2024-02-12 09:50:00 | 544.59 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-02-15 10:00:00 | 542.15 | 2024-02-15 10:25:00 | 543.84 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-02-19 09:55:00 | 540.10 | 2024-02-19 10:00:00 | 541.73 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-20 10:45:00 | 536.80 | 2024-02-20 11:00:00 | 538.17 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-02-21 09:30:00 | 537.00 | 2024-02-21 10:00:00 | 538.26 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-02-26 09:45:00 | 531.25 | 2024-02-26 09:50:00 | 532.53 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-02-29 09:30:00 | 529.45 | 2024-02-29 10:05:00 | 526.71 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-02-29 09:30:00 | 529.45 | 2024-02-29 10:55:00 | 529.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 10:45:00 | 532.60 | 2024-03-01 11:40:00 | 534.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-03-05 09:50:00 | 529.05 | 2024-03-05 10:35:00 | 530.94 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-03-06 10:10:00 | 528.30 | 2024-03-06 10:20:00 | 529.91 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-11 09:30:00 | 517.25 | 2024-03-11 11:55:00 | 513.76 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-03-11 09:30:00 | 517.25 | 2024-03-11 12:05:00 | 517.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-12 10:25:00 | 518.50 | 2024-03-12 10:40:00 | 520.06 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-13 09:40:00 | 517.35 | 2024-03-13 09:50:00 | 514.71 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-03-13 09:40:00 | 517.35 | 2024-03-13 11:35:00 | 514.25 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-03-15 09:35:00 | 515.30 | 2024-03-15 09:45:00 | 517.43 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-03-19 09:30:00 | 504.05 | 2024-03-19 09:35:00 | 505.97 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-03-20 09:55:00 | 520.10 | 2024-03-20 10:00:00 | 517.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-03-22 10:10:00 | 524.85 | 2024-03-22 10:20:00 | 527.46 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-03-22 10:10:00 | 524.85 | 2024-03-22 10:30:00 | 524.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-01 09:40:00 | 544.30 | 2024-04-01 10:35:00 | 547.01 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-04-01 09:40:00 | 544.30 | 2024-04-01 10:55:00 | 544.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-16 09:55:00 | 582.20 | 2024-04-16 10:00:00 | 584.01 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-04-18 09:45:00 | 585.30 | 2024-04-18 13:25:00 | 582.14 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-04-18 09:45:00 | 585.30 | 2024-04-18 15:20:00 | 574.60 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2024-04-23 10:55:00 | 581.95 | 2024-04-23 11:25:00 | 585.25 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-04-23 10:55:00 | 581.95 | 2024-04-23 12:45:00 | 584.30 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2024-04-24 09:30:00 | 615.75 | 2024-04-24 09:45:00 | 611.77 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest1 | 2024-04-25 10:50:00 | 599.00 | 2024-04-25 11:55:00 | 595.77 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-04-25 10:50:00 | 599.00 | 2024-04-25 14:45:00 | 599.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-29 10:00:00 | 615.00 | 2024-04-29 10:15:00 | 612.47 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-03 11:00:00 | 599.40 | 2024-05-03 12:40:00 | 601.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-06 10:25:00 | 623.10 | 2024-05-06 10:30:00 | 628.66 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2024-05-06 10:25:00 | 623.10 | 2024-05-06 11:00:00 | 639.95 | TARGET_HIT | 0.50 | 2.70% |
