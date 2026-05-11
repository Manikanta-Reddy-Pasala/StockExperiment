# IIFL Finance Ltd. (IIFL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 460.10
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
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 12 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 74
- **Target hits / Stop hits / Partials:** 12 / 74 / 28
- **Avg / median % per leg:** 0.02% / -0.20%
- **Sum % (uncompounded):** 1.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 25 | 38.5% | 7 | 40 | 18 | 0.06% | 3.7% |
| BUY @ 2nd Alert (retest1) | 65 | 25 | 38.5% | 7 | 40 | 18 | 0.06% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 15 | 30.6% | 5 | 34 | 10 | -0.04% | -1.9% |
| SELL @ 2nd Alert (retest1) | 49 | 15 | 30.6% | 5 | 34 | 10 | -0.04% | -1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 114 | 40 | 35.1% | 12 | 74 | 28 | 0.02% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 11:10:00 | 415.05 | 410.04 | 0.00 | ORB-long ORB[406.20,409.95] vol=2.1x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 11:35:00 | 417.06 | 411.09 | 0.00 | T1 1.5R @ 417.06 |
| Stop hit — per-position SL triggered | 2025-05-14 11:40:00 | 415.05 | 411.32 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:40:00 | 417.05 | 414.67 | 0.00 | ORB-long ORB[410.60,415.45] vol=1.7x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-05-23 09:55:00 | 415.62 | 415.71 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:10:00 | 420.50 | 418.54 | 0.00 | ORB-long ORB[415.40,419.30] vol=2.2x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:40:00 | 422.21 | 419.24 | 0.00 | T1 1.5R @ 422.21 |
| Target hit | 2025-05-28 13:45:00 | 421.35 | 422.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2025-05-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:40:00 | 426.50 | 424.34 | 0.00 | ORB-long ORB[422.50,425.40] vol=1.7x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 10:20:00 | 428.41 | 426.53 | 0.00 | T1 1.5R @ 428.41 |
| Target hit | 2025-05-29 13:35:00 | 427.55 | 428.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2025-06-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:10:00 | 426.20 | 429.28 | 0.00 | ORB-short ORB[429.10,433.00] vol=2.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-06-03 11:40:00 | 427.24 | 428.75 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:45:00 | 431.95 | 430.19 | 0.00 | ORB-long ORB[428.00,431.40] vol=2.7x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-06-05 11:10:00 | 430.65 | 430.30 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 434.80 | 431.30 | 0.00 | ORB-long ORB[429.30,434.60] vol=2.1x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 433.14 | 431.59 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:05:00 | 482.00 | 487.42 | 0.00 | ORB-short ORB[486.00,491.95] vol=1.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-06-10 11:30:00 | 483.88 | 487.28 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 11:10:00 | 492.30 | 486.32 | 0.00 | ORB-long ORB[480.25,486.45] vol=2.7x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 11:25:00 | 494.65 | 488.05 | 0.00 | T1 1.5R @ 494.65 |
| Stop hit — per-position SL triggered | 2025-06-11 12:05:00 | 492.30 | 489.21 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 487.90 | 485.75 | 0.00 | ORB-long ORB[482.05,486.25] vol=1.7x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 10:15:00 | 490.65 | 487.88 | 0.00 | T1 1.5R @ 490.65 |
| Target hit | 2025-06-17 12:00:00 | 490.10 | 490.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2025-06-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:30:00 | 478.55 | 483.67 | 0.00 | ORB-short ORB[480.90,487.95] vol=1.8x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-06-19 10:55:00 | 480.26 | 482.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:30:00 | 471.60 | 474.66 | 0.00 | ORB-short ORB[473.60,480.20] vol=1.7x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-06-20 09:35:00 | 473.62 | 474.50 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:00:00 | 478.10 | 473.77 | 0.00 | ORB-long ORB[471.05,475.20] vol=4.9x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 12:40:00 | 481.36 | 476.57 | 0.00 | T1 1.5R @ 481.36 |
| Stop hit — per-position SL triggered | 2025-06-23 14:50:00 | 478.10 | 478.10 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 09:35:00 | 478.30 | 479.78 | 0.00 | ORB-short ORB[479.00,482.50] vol=1.9x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-06-25 09:40:00 | 479.71 | 479.67 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 09:50:00 | 482.75 | 482.80 | 0.00 | ORB-short ORB[483.00,485.80] vol=2.7x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:00:00 | 479.93 | 482.48 | 0.00 | T1 1.5R @ 479.93 |
| Target hit | 2025-06-26 12:20:00 | 478.50 | 478.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2025-07-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 09:55:00 | 471.10 | 471.39 | 0.00 | ORB-short ORB[471.55,475.35] vol=6.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-07-01 10:25:00 | 472.71 | 471.44 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:30:00 | 512.05 | 509.89 | 0.00 | ORB-long ORB[505.25,511.15] vol=1.6x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-07-10 09:40:00 | 510.45 | 510.39 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:35:00 | 530.35 | 528.25 | 0.00 | ORB-long ORB[524.00,529.90] vol=2.1x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-07-15 09:45:00 | 528.77 | 528.53 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:50:00 | 528.65 | 525.95 | 0.00 | ORB-long ORB[523.55,527.70] vol=2.4x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-07-16 09:55:00 | 527.20 | 526.04 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:35:00 | 524.55 | 526.15 | 0.00 | ORB-short ORB[526.05,529.35] vol=1.6x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-07-17 10:25:00 | 525.81 | 525.22 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:55:00 | 534.20 | 532.37 | 0.00 | ORB-long ORB[526.00,532.90] vol=8.0x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-07-18 10:05:00 | 532.74 | 532.52 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:50:00 | 530.50 | 532.76 | 0.00 | ORB-short ORB[531.30,536.40] vol=1.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-07-22 10:00:00 | 531.89 | 532.69 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:35:00 | 525.25 | 526.22 | 0.00 | ORB-short ORB[525.85,530.00] vol=1.6x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:40:00 | 523.11 | 525.75 | 0.00 | T1 1.5R @ 523.11 |
| Stop hit — per-position SL triggered | 2025-07-23 09:45:00 | 525.25 | 525.73 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:10:00 | 531.25 | 533.50 | 0.00 | ORB-short ORB[533.05,537.85] vol=2.8x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-07-24 10:20:00 | 532.69 | 533.34 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:35:00 | 518.10 | 523.13 | 0.00 | ORB-short ORB[523.15,528.00] vol=2.2x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-07-25 09:40:00 | 520.10 | 522.59 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 11:15:00 | 452.95 | 451.05 | 0.00 | ORB-long ORB[449.05,452.60] vol=2.1x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-08-08 12:20:00 | 451.62 | 451.46 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:55:00 | 446.75 | 448.20 | 0.00 | ORB-short ORB[449.60,452.05] vol=1.8x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-08-13 11:00:00 | 447.90 | 448.13 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:40:00 | 449.60 | 447.23 | 0.00 | ORB-long ORB[445.00,448.30] vol=1.7x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-08-18 09:45:00 | 448.02 | 447.53 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:50:00 | 457.70 | 452.41 | 0.00 | ORB-long ORB[446.55,448.95] vol=2.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-08-19 10:55:00 | 456.36 | 453.06 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:45:00 | 442.95 | 440.88 | 0.00 | ORB-long ORB[437.50,442.00] vol=1.8x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:00:00 | 444.88 | 441.61 | 0.00 | T1 1.5R @ 444.88 |
| Target hit | 2025-09-02 11:05:00 | 443.05 | 443.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:15:00 | 432.90 | 435.51 | 0.00 | ORB-short ORB[433.05,436.25] vol=2.0x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 434.17 | 435.36 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:25:00 | 437.60 | 436.45 | 0.00 | ORB-long ORB[432.65,437.20] vol=2.2x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-09-08 10:35:00 | 436.34 | 436.47 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:00:00 | 448.40 | 444.66 | 0.00 | ORB-long ORB[439.00,445.35] vol=2.2x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-09-10 11:20:00 | 447.17 | 445.12 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 09:30:00 | 443.65 | 445.94 | 0.00 | ORB-short ORB[444.55,449.15] vol=2.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-09-11 09:35:00 | 444.83 | 445.82 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:00:00 | 438.35 | 439.65 | 0.00 | ORB-short ORB[438.70,442.20] vol=1.7x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:15:00 | 436.94 | 439.42 | 0.00 | T1 1.5R @ 436.94 |
| Stop hit — per-position SL triggered | 2025-09-12 11:35:00 | 438.35 | 439.10 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 11:15:00 | 438.95 | 436.94 | 0.00 | ORB-long ORB[434.00,437.35] vol=5.0x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-09-15 11:20:00 | 438.06 | 436.95 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:30:00 | 448.55 | 446.38 | 0.00 | ORB-long ORB[440.30,443.80] vol=1.5x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 11:00:00 | 450.58 | 447.51 | 0.00 | T1 1.5R @ 450.58 |
| Target hit | 2025-09-16 12:05:00 | 449.80 | 449.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 443.65 | 445.63 | 0.00 | ORB-short ORB[445.60,448.65] vol=2.2x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-09-17 09:50:00 | 444.91 | 445.29 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:30:00 | 450.50 | 448.02 | 0.00 | ORB-long ORB[444.90,448.00] vol=3.8x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-09-18 09:40:00 | 449.13 | 448.72 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:35:00 | 447.60 | 449.33 | 0.00 | ORB-short ORB[450.55,453.85] vol=2.1x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:40:00 | 445.15 | 449.01 | 0.00 | T1 1.5R @ 445.15 |
| Stop hit — per-position SL triggered | 2025-09-23 10:05:00 | 447.60 | 447.16 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:30:00 | 447.70 | 449.50 | 0.00 | ORB-short ORB[448.00,451.35] vol=2.4x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:55:00 | 445.53 | 448.90 | 0.00 | T1 1.5R @ 445.53 |
| Target hit | 2025-09-24 13:50:00 | 447.00 | 446.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 429.05 | 431.02 | 0.00 | ORB-short ORB[429.55,433.90] vol=2.8x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:00:00 | 426.97 | 428.84 | 0.00 | T1 1.5R @ 426.97 |
| Target hit | 2025-09-26 10:35:00 | 428.50 | 428.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — BUY (started 2025-10-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:20:00 | 460.65 | 457.67 | 0.00 | ORB-long ORB[455.05,459.10] vol=1.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-10-06 10:30:00 | 458.98 | 457.90 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 479.40 | 475.21 | 0.00 | ORB-long ORB[470.75,476.70] vol=2.0x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:35:00 | 482.25 | 477.06 | 0.00 | T1 1.5R @ 482.25 |
| Stop hit — per-position SL triggered | 2025-10-07 09:55:00 | 479.40 | 480.45 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:50:00 | 494.30 | 491.04 | 0.00 | ORB-long ORB[485.25,491.85] vol=2.7x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-10-15 09:55:00 | 492.34 | 491.19 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:10:00 | 502.00 | 503.78 | 0.00 | ORB-short ORB[503.50,509.80] vol=3.3x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 503.36 | 503.85 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:55:00 | 496.00 | 498.86 | 0.00 | ORB-short ORB[498.75,505.85] vol=1.8x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-10-17 11:10:00 | 497.69 | 498.70 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 500.75 | 495.93 | 0.00 | ORB-long ORB[491.00,493.95] vol=7.0x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-10-27 09:40:00 | 499.06 | 497.21 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:00:00 | 512.25 | 508.83 | 0.00 | ORB-long ORB[506.00,510.40] vol=2.4x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-10-28 10:25:00 | 510.55 | 509.25 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:45:00 | 527.35 | 521.20 | 0.00 | ORB-long ORB[516.70,522.70] vol=1.5x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-11-07 10:55:00 | 525.30 | 522.33 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 530.95 | 528.29 | 0.00 | ORB-long ORB[524.50,529.65] vol=2.3x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:45:00 | 534.08 | 530.73 | 0.00 | T1 1.5R @ 534.08 |
| Stop hit — per-position SL triggered | 2025-11-10 10:00:00 | 530.95 | 531.16 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:35:00 | 547.90 | 545.11 | 0.00 | ORB-long ORB[542.05,547.30] vol=1.8x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-11-11 09:45:00 | 546.16 | 545.82 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:15:00 | 535.30 | 539.38 | 0.00 | ORB-short ORB[538.75,544.80] vol=1.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-11-12 10:45:00 | 536.83 | 538.60 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:00:00 | 553.80 | 550.37 | 0.00 | ORB-long ORB[545.30,551.80] vol=1.8x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-11-17 10:40:00 | 551.75 | 551.52 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:35:00 | 551.15 | 554.23 | 0.00 | ORB-short ORB[551.20,559.10] vol=1.9x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-11-20 09:40:00 | 553.54 | 553.45 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:45:00 | 540.80 | 542.89 | 0.00 | ORB-short ORB[542.10,546.00] vol=1.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-11-21 11:55:00 | 542.22 | 542.23 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 574.15 | 571.25 | 0.00 | ORB-long ORB[568.15,573.30] vol=3.4x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:10:00 | 576.66 | 572.82 | 0.00 | T1 1.5R @ 576.66 |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 574.15 | 572.97 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:10:00 | 572.85 | 576.28 | 0.00 | ORB-short ORB[575.50,582.25] vol=3.0x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:55:00 | 569.23 | 575.29 | 0.00 | T1 1.5R @ 569.23 |
| Stop hit — per-position SL triggered | 2025-12-03 11:30:00 | 572.85 | 574.61 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:30:00 | 578.95 | 574.62 | 0.00 | ORB-long ORB[571.05,575.65] vol=1.5x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-12-04 09:35:00 | 577.04 | 575.37 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:50:00 | 558.75 | 562.74 | 0.00 | ORB-short ORB[561.00,567.25] vol=1.9x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 560.68 | 562.47 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 573.85 | 574.01 | 0.00 | ORB-short ORB[575.40,581.00] vol=2.0x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-12-15 13:35:00 | 575.31 | 573.89 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:10:00 | 574.35 | 572.08 | 0.00 | ORB-long ORB[567.50,574.00] vol=2.2x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-12-23 10:20:00 | 572.38 | 572.16 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:00:00 | 595.20 | 601.87 | 0.00 | ORB-short ORB[602.70,607.55] vol=2.2x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-12-26 11:10:00 | 597.00 | 601.47 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:55:00 | 593.00 | 596.74 | 0.00 | ORB-short ORB[593.95,602.65] vol=1.7x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-12-29 10:05:00 | 595.18 | 596.23 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 11:10:00 | 597.25 | 593.85 | 0.00 | ORB-long ORB[591.00,595.25] vol=3.1x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:25:00 | 599.55 | 595.26 | 0.00 | T1 1.5R @ 599.55 |
| Stop hit — per-position SL triggered | 2025-12-30 11:55:00 | 597.25 | 596.06 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:45:00 | 607.45 | 602.76 | 0.00 | ORB-long ORB[597.95,601.90] vol=1.9x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-12-31 10:50:00 | 605.35 | 603.34 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:50:00 | 613.90 | 613.24 | 0.00 | ORB-long ORB[609.45,612.50] vol=3.5x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:10:00 | 616.90 | 613.78 | 0.00 | T1 1.5R @ 616.90 |
| Target hit | 2026-01-01 14:15:00 | 615.50 | 616.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — SELL (started 2026-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:05:00 | 653.95 | 660.32 | 0.00 | ORB-short ORB[659.35,665.50] vol=2.2x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 656.42 | 659.63 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 644.05 | 649.77 | 0.00 | ORB-short ORB[648.45,657.00] vol=2.8x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:30:00 | 640.78 | 648.70 | 0.00 | T1 1.5R @ 640.78 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 644.05 | 648.64 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 09:35:00 | 615.10 | 621.83 | 0.00 | ORB-short ORB[620.10,629.35] vol=1.8x ATR=3.55 |
| Stop hit — per-position SL triggered | 2026-01-21 09:40:00 | 618.65 | 621.00 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:35:00 | 522.05 | 518.00 | 0.00 | ORB-long ORB[511.35,518.40] vol=2.0x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:45:00 | 526.37 | 519.80 | 0.00 | T1 1.5R @ 526.37 |
| Stop hit — per-position SL triggered | 2026-02-04 12:35:00 | 522.05 | 525.24 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 513.70 | 510.13 | 0.00 | ORB-long ORB[506.55,511.40] vol=2.2x ATR=1.91 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 511.79 | 510.53 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 511.75 | 517.46 | 0.00 | ORB-short ORB[517.25,523.00] vol=2.2x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:15:00 | 508.37 | 515.28 | 0.00 | T1 1.5R @ 508.37 |
| Target hit | 2026-02-18 13:35:00 | 507.50 | 507.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 504.00 | 498.93 | 0.00 | ORB-long ORB[496.45,503.00] vol=1.9x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-02-24 10:20:00 | 501.66 | 500.55 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 503.25 | 506.56 | 0.00 | ORB-short ORB[505.15,511.90] vol=4.1x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-02-26 10:30:00 | 504.92 | 505.64 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 496.60 | 497.48 | 0.00 | ORB-short ORB[497.50,504.90] vol=1.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-02-27 15:00:00 | 498.37 | 497.09 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 489.25 | 484.46 | 0.00 | ORB-long ORB[479.10,486.00] vol=2.8x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 487.82 | 485.38 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 489.00 | 492.15 | 0.00 | ORB-short ORB[490.10,497.45] vol=5.2x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-03-06 09:50:00 | 491.28 | 492.02 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 479.75 | 475.83 | 0.00 | ORB-long ORB[473.50,479.00] vol=2.2x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:30:00 | 482.23 | 476.49 | 0.00 | T1 1.5R @ 482.23 |
| Target hit | 2026-03-10 15:20:00 | 492.50 | 483.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 479.15 | 476.56 | 0.00 | ORB-long ORB[472.50,477.50] vol=1.5x ATR=2.76 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 476.39 | 476.78 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:00:00 | 452.90 | 453.02 | 0.00 | ORB-short ORB[453.70,459.65] vol=2.0x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 449.49 | 452.43 | 0.00 | T1 1.5R @ 449.49 |
| Target hit | 2026-03-27 12:25:00 | 450.30 | 450.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 82 — SELL (started 2026-04-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:50:00 | 430.00 | 431.68 | 0.00 | ORB-short ORB[431.55,437.45] vol=2.4x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-04-07 10:00:00 | 432.27 | 431.75 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:55:00 | 452.00 | 445.07 | 0.00 | ORB-long ORB[442.25,448.00] vol=2.0x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-04-08 11:00:00 | 449.42 | 445.35 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:55:00 | 450.50 | 445.57 | 0.00 | ORB-long ORB[440.55,446.10] vol=2.2x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:20:00 | 453.19 | 446.41 | 0.00 | T1 1.5R @ 453.19 |
| Stop hit — per-position SL triggered | 2026-04-13 11:30:00 | 450.50 | 446.77 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 474.60 | 471.77 | 0.00 | ORB-long ORB[467.45,473.00] vol=1.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 477.69 | 472.84 | 0.00 | T1 1.5R @ 477.69 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 474.60 | 473.93 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 468.00 | 464.86 | 0.00 | ORB-long ORB[460.25,467.00] vol=2.3x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:35:00 | 470.32 | 466.28 | 0.00 | T1 1.5R @ 470.32 |
| Stop hit — per-position SL triggered | 2026-05-08 09:40:00 | 468.00 | 466.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 11:10:00 | 415.05 | 2025-05-14 11:35:00 | 417.06 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-14 11:10:00 | 415.05 | 2025-05-14 11:40:00 | 415.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 09:40:00 | 417.05 | 2025-05-23 09:55:00 | 415.62 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-28 10:10:00 | 420.50 | 2025-05-28 10:40:00 | 422.21 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-05-28 10:10:00 | 420.50 | 2025-05-28 13:45:00 | 421.35 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-05-29 09:40:00 | 426.50 | 2025-05-29 10:20:00 | 428.41 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-29 09:40:00 | 426.50 | 2025-05-29 13:35:00 | 427.55 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-06-03 11:10:00 | 426.20 | 2025-06-03 11:40:00 | 427.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-05 10:45:00 | 431.95 | 2025-06-05 11:10:00 | 430.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-06 10:05:00 | 434.80 | 2025-06-06 10:10:00 | 433.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-06-10 11:05:00 | 482.00 | 2025-06-10 11:30:00 | 483.88 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-11 11:10:00 | 492.30 | 2025-06-11 11:25:00 | 494.65 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-06-11 11:10:00 | 492.30 | 2025-06-11 12:05:00 | 492.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-17 09:30:00 | 487.90 | 2025-06-17 10:15:00 | 490.65 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-17 09:30:00 | 487.90 | 2025-06-17 12:00:00 | 490.10 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-19 10:30:00 | 478.55 | 2025-06-19 10:55:00 | 480.26 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-06-20 09:30:00 | 471.60 | 2025-06-20 09:35:00 | 473.62 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-23 11:00:00 | 478.10 | 2025-06-23 12:40:00 | 481.36 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-06-23 11:00:00 | 478.10 | 2025-06-23 14:50:00 | 478.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-25 09:35:00 | 478.30 | 2025-06-25 09:40:00 | 479.71 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-26 09:50:00 | 482.75 | 2025-06-26 10:00:00 | 479.93 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-06-26 09:50:00 | 482.75 | 2025-06-26 12:20:00 | 478.50 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-07-01 09:55:00 | 471.10 | 2025-07-01 10:25:00 | 472.71 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-10 09:30:00 | 512.05 | 2025-07-10 09:40:00 | 510.45 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-15 09:35:00 | 530.35 | 2025-07-15 09:45:00 | 528.77 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-16 09:50:00 | 528.65 | 2025-07-16 09:55:00 | 527.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-17 09:35:00 | 524.55 | 2025-07-17 10:25:00 | 525.81 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-18 09:55:00 | 534.20 | 2025-07-18 10:05:00 | 532.74 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 09:50:00 | 530.50 | 2025-07-22 10:00:00 | 531.89 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-23 09:35:00 | 525.25 | 2025-07-23 09:40:00 | 523.11 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-23 09:35:00 | 525.25 | 2025-07-23 09:45:00 | 525.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 10:10:00 | 531.25 | 2025-07-24 10:20:00 | 532.69 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-25 09:35:00 | 518.10 | 2025-07-25 09:40:00 | 520.10 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-08-08 11:15:00 | 452.95 | 2025-08-08 12:20:00 | 451.62 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-13 10:55:00 | 446.75 | 2025-08-13 11:00:00 | 447.90 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-18 09:40:00 | 449.60 | 2025-08-18 09:45:00 | 448.02 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-19 10:50:00 | 457.70 | 2025-08-19 10:55:00 | 456.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-02 09:45:00 | 442.95 | 2025-09-02 10:00:00 | 444.88 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-02 09:45:00 | 442.95 | 2025-09-02 11:05:00 | 443.05 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2025-09-05 10:15:00 | 432.90 | 2025-09-05 10:20:00 | 434.17 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-08 10:25:00 | 437.60 | 2025-09-08 10:35:00 | 436.34 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-10 11:00:00 | 448.40 | 2025-09-10 11:20:00 | 447.17 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-11 09:30:00 | 443.65 | 2025-09-11 09:35:00 | 444.83 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-12 11:00:00 | 438.35 | 2025-09-12 11:15:00 | 436.94 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-09-12 11:00:00 | 438.35 | 2025-09-12 11:35:00 | 438.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-15 11:15:00 | 438.95 | 2025-09-15 11:20:00 | 438.06 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-16 10:30:00 | 448.55 | 2025-09-16 11:00:00 | 450.58 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-16 10:30:00 | 448.55 | 2025-09-16 12:05:00 | 449.80 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2025-09-17 09:45:00 | 443.65 | 2025-09-17 09:50:00 | 444.91 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-18 09:30:00 | 450.50 | 2025-09-18 09:40:00 | 449.13 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-23 09:35:00 | 447.60 | 2025-09-23 09:40:00 | 445.15 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-09-23 09:35:00 | 447.60 | 2025-09-23 10:05:00 | 447.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 10:30:00 | 447.70 | 2025-09-24 10:55:00 | 445.53 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-09-24 10:30:00 | 447.70 | 2025-09-24 13:50:00 | 447.00 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-09-26 09:30:00 | 429.05 | 2025-09-26 10:00:00 | 426.97 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-09-26 09:30:00 | 429.05 | 2025-09-26 10:35:00 | 428.50 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2025-10-06 10:20:00 | 460.65 | 2025-10-06 10:30:00 | 458.98 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-07 09:30:00 | 479.40 | 2025-10-07 09:35:00 | 482.25 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-10-07 09:30:00 | 479.40 | 2025-10-07 09:55:00 | 479.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 09:50:00 | 494.30 | 2025-10-15 09:55:00 | 492.34 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-16 10:10:00 | 502.00 | 2025-10-16 10:15:00 | 503.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-17 10:55:00 | 496.00 | 2025-10-17 11:10:00 | 497.69 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-27 09:30:00 | 500.75 | 2025-10-27 09:40:00 | 499.06 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-28 10:00:00 | 512.25 | 2025-10-28 10:25:00 | 510.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-07 10:45:00 | 527.35 | 2025-11-07 10:55:00 | 525.30 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-11-10 09:30:00 | 530.95 | 2025-11-10 09:45:00 | 534.08 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-11-10 09:30:00 | 530.95 | 2025-11-10 10:00:00 | 530.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-11 09:35:00 | 547.90 | 2025-11-11 09:45:00 | 546.16 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-12 10:15:00 | 535.30 | 2025-11-12 10:45:00 | 536.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-17 10:00:00 | 553.80 | 2025-11-17 10:40:00 | 551.75 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-20 09:35:00 | 551.15 | 2025-11-20 09:40:00 | 553.54 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-11-21 10:45:00 | 540.80 | 2025-11-21 11:55:00 | 542.22 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-28 09:45:00 | 574.15 | 2025-11-28 10:10:00 | 576.66 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-28 09:45:00 | 574.15 | 2025-11-28 10:15:00 | 574.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 10:10:00 | 572.85 | 2025-12-03 10:55:00 | 569.23 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-12-03 10:10:00 | 572.85 | 2025-12-03 11:30:00 | 572.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 09:30:00 | 578.95 | 2025-12-04 09:35:00 | 577.04 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-05 09:50:00 | 558.75 | 2025-12-05 10:00:00 | 560.68 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-15 11:15:00 | 573.85 | 2025-12-15 13:35:00 | 575.31 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-23 10:10:00 | 574.35 | 2025-12-23 10:20:00 | 572.38 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-26 11:00:00 | 595.20 | 2025-12-26 11:10:00 | 597.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-29 09:55:00 | 593.00 | 2025-12-29 10:05:00 | 595.18 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-30 11:10:00 | 597.25 | 2025-12-30 11:25:00 | 599.55 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-30 11:10:00 | 597.25 | 2025-12-30 11:55:00 | 597.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:45:00 | 607.45 | 2025-12-31 10:50:00 | 605.35 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-01-01 10:50:00 | 613.90 | 2026-01-01 11:10:00 | 616.90 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-01-01 10:50:00 | 613.90 | 2026-01-01 14:15:00 | 615.50 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2026-01-07 10:05:00 | 653.95 | 2026-01-07 10:15:00 | 656.42 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-01-08 11:10:00 | 644.05 | 2026-01-08 11:30:00 | 640.78 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-01-08 11:10:00 | 644.05 | 2026-01-08 11:35:00 | 644.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-21 09:35:00 | 615.10 | 2026-01-21 09:40:00 | 618.65 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-02-04 09:35:00 | 522.05 | 2026-02-04 09:45:00 | 526.37 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2026-02-04 09:35:00 | 522.05 | 2026-02-04 12:35:00 | 522.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:25:00 | 513.70 | 2026-02-17 10:40:00 | 511.79 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 10:55:00 | 511.75 | 2026-02-18 11:15:00 | 508.37 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-02-18 10:55:00 | 511.75 | 2026-02-18 13:35:00 | 507.50 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2026-02-24 09:45:00 | 504.00 | 2026-02-24 10:20:00 | 501.66 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-26 10:15:00 | 503.25 | 2026-02-26 10:30:00 | 504.92 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-27 10:45:00 | 496.60 | 2026-02-27 15:00:00 | 498.37 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-05 11:15:00 | 489.25 | 2026-03-05 11:45:00 | 487.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 09:45:00 | 489.00 | 2026-03-06 09:50:00 | 491.28 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-10 11:10:00 | 479.75 | 2026-03-10 11:30:00 | 482.23 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-10 11:10:00 | 479.75 | 2026-03-10 15:20:00 | 492.50 | TARGET_HIT | 0.50 | 2.66% |
| BUY | retest1 | 2026-03-20 09:30:00 | 479.15 | 2026-03-20 09:50:00 | 476.39 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-03-27 10:00:00 | 452.90 | 2026-03-27 10:15:00 | 449.49 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2026-03-27 10:00:00 | 452.90 | 2026-03-27 12:25:00 | 450.30 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-07 09:50:00 | 430.00 | 2026-04-07 10:00:00 | 432.27 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-08 10:55:00 | 452.00 | 2026-04-08 11:00:00 | 449.42 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-13 10:55:00 | 450.50 | 2026-04-13 11:20:00 | 453.19 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-13 10:55:00 | 450.50 | 2026-04-13 11:30:00 | 450.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:30:00 | 474.60 | 2026-04-21 09:35:00 | 477.69 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-21 09:30:00 | 474.60 | 2026-04-21 09:50:00 | 474.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:30:00 | 468.00 | 2026-05-08 09:35:00 | 470.32 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-08 09:30:00 | 468.00 | 2026-05-08 09:40:00 | 468.00 | STOP_HIT | 0.50 | 0.00% |
