# Star Health and Allied Insurance Company Ltd. (STARHEALTH)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 519.05
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
| PARTIAL | 34 |
| TARGET_HIT | 17 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 58
- **Target hits / Stop hits / Partials:** 17 / 58 / 34
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 19.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 30 | 46.9% | 10 | 34 | 20 | 0.22% | 14.0% |
| BUY @ 2nd Alert (retest1) | 64 | 30 | 46.9% | 10 | 34 | 20 | 0.22% | 14.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 45 | 21 | 46.7% | 7 | 24 | 14 | 0.12% | 5.6% |
| SELL @ 2nd Alert (retest1) | 45 | 21 | 46.7% | 7 | 24 | 14 | 0.12% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 109 | 51 | 46.8% | 17 | 58 | 34 | 0.18% | 19.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:35:00 | 443.75 | 438.28 | 0.00 | ORB-long ORB[432.30,438.30] vol=2.1x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-05-21 10:55:00 | 441.93 | 438.70 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:10:00 | 447.60 | 446.58 | 0.00 | ORB-long ORB[440.70,445.90] vol=1.6x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 11:00:00 | 450.01 | 448.46 | 0.00 | T1 1.5R @ 450.01 |
| Target hit | 2025-05-23 15:20:00 | 462.30 | 454.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-05-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:40:00 | 470.65 | 469.36 | 0.00 | ORB-long ORB[464.40,469.30] vol=3.7x ATR=2.29 |
| Stop hit — per-position SL triggered | 2025-05-28 10:00:00 | 468.36 | 469.94 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 10:35:00 | 473.90 | 476.38 | 0.00 | ORB-short ORB[474.30,479.90] vol=3.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-06-02 11:40:00 | 475.75 | 475.55 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:05:00 | 472.35 | 475.70 | 0.00 | ORB-short ORB[472.60,477.00] vol=1.6x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 12:00:00 | 470.63 | 475.04 | 0.00 | T1 1.5R @ 470.63 |
| Target hit | 2025-06-04 15:20:00 | 466.95 | 471.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 10:10:00 | 438.95 | 442.37 | 0.00 | ORB-short ORB[443.40,446.90] vol=2.5x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 440.34 | 441.22 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:15:00 | 430.60 | 432.93 | 0.00 | ORB-short ORB[431.60,436.45] vol=2.5x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:40:00 | 428.37 | 431.25 | 0.00 | T1 1.5R @ 428.37 |
| Target hit | 2025-06-19 15:05:00 | 430.15 | 430.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:15:00 | 434.80 | 429.74 | 0.00 | ORB-long ORB[427.00,431.85] vol=19.8x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:20:00 | 438.68 | 429.91 | 0.00 | T1 1.5R @ 438.68 |
| Stop hit — per-position SL triggered | 2025-06-25 10:25:00 | 434.80 | 430.13 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:15:00 | 414.00 | 417.43 | 0.00 | ORB-short ORB[418.85,425.00] vol=2.6x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-07-01 10:20:00 | 415.48 | 417.33 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 10:50:00 | 416.25 | 417.32 | 0.00 | ORB-short ORB[417.15,419.95] vol=5.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-07-03 12:10:00 | 417.52 | 416.80 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:45:00 | 426.90 | 425.00 | 0.00 | ORB-long ORB[421.00,425.25] vol=4.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-07-08 09:50:00 | 425.63 | 425.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:00:00 | 430.40 | 427.81 | 0.00 | ORB-long ORB[425.00,428.90] vol=2.5x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 11:20:00 | 432.11 | 428.74 | 0.00 | T1 1.5R @ 432.11 |
| Target hit | 2025-07-09 15:10:00 | 431.15 | 432.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2025-07-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:40:00 | 428.70 | 431.86 | 0.00 | ORB-short ORB[431.05,434.70] vol=1.8x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 12:05:00 | 426.81 | 430.50 | 0.00 | T1 1.5R @ 426.81 |
| Target hit | 2025-07-10 15:20:00 | 427.10 | 428.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-07-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:25:00 | 423.25 | 425.21 | 0.00 | ORB-short ORB[424.00,428.00] vol=1.5x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:40:00 | 421.53 | 424.68 | 0.00 | T1 1.5R @ 421.53 |
| Target hit | 2025-07-11 13:05:00 | 421.20 | 420.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 435.20 | 433.35 | 0.00 | ORB-long ORB[428.55,433.50] vol=5.7x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 09:55:00 | 437.80 | 434.77 | 0.00 | T1 1.5R @ 437.80 |
| Target hit | 2025-07-16 11:40:00 | 440.05 | 440.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-07-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:05:00 | 439.70 | 441.10 | 0.00 | ORB-short ORB[440.60,444.20] vol=2.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 440.76 | 441.08 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:45:00 | 420.00 | 422.17 | 0.00 | ORB-short ORB[420.05,425.00] vol=2.3x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-07-29 11:00:00 | 421.38 | 421.92 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 11:00:00 | 433.50 | 430.00 | 0.00 | ORB-long ORB[426.55,432.55] vol=2.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-08-07 11:40:00 | 432.05 | 431.13 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:30:00 | 440.40 | 438.35 | 0.00 | ORB-long ORB[433.95,439.50] vol=1.9x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-08-11 09:40:00 | 438.64 | 438.35 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:05:00 | 441.40 | 443.20 | 0.00 | ORB-short ORB[443.15,447.55] vol=2.2x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-08-14 10:10:00 | 443.05 | 443.94 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:00:00 | 443.00 | 445.29 | 0.00 | ORB-short ORB[443.65,450.05] vol=1.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:15:00 | 440.98 | 445.08 | 0.00 | T1 1.5R @ 440.98 |
| Stop hit — per-position SL triggered | 2025-08-19 11:25:00 | 443.00 | 445.06 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:15:00 | 443.50 | 441.52 | 0.00 | ORB-long ORB[437.40,442.45] vol=2.2x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-08-22 12:00:00 | 441.90 | 442.14 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:30:00 | 438.80 | 443.03 | 0.00 | ORB-short ORB[442.05,447.20] vol=2.1x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-08-25 12:00:00 | 440.41 | 441.39 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:05:00 | 449.05 | 445.19 | 0.00 | ORB-long ORB[441.05,444.40] vol=2.0x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-09-03 10:10:00 | 447.70 | 445.49 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:30:00 | 454.10 | 451.62 | 0.00 | ORB-long ORB[449.00,452.95] vol=2.4x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-09-24 09:35:00 | 452.58 | 451.92 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:40:00 | 457.70 | 453.89 | 0.00 | ORB-long ORB[451.10,456.30] vol=1.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-09-25 11:50:00 | 456.37 | 455.20 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 447.60 | 449.60 | 0.00 | ORB-short ORB[448.90,454.90] vol=3.1x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-09-26 09:40:00 | 449.40 | 449.48 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 09:35:00 | 455.30 | 458.46 | 0.00 | ORB-short ORB[457.00,461.50] vol=2.0x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-10-07 10:00:00 | 456.71 | 458.00 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:15:00 | 479.05 | 476.98 | 0.00 | ORB-long ORB[473.80,477.30] vol=1.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-10-08 10:30:00 | 477.10 | 477.14 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 09:55:00 | 475.65 | 477.70 | 0.00 | ORB-short ORB[476.10,481.00] vol=1.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-10-10 10:05:00 | 477.53 | 477.61 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:35:00 | 472.95 | 474.88 | 0.00 | ORB-short ORB[473.05,479.65] vol=2.0x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-10-14 09:45:00 | 474.28 | 474.58 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:40:00 | 487.40 | 484.86 | 0.00 | ORB-long ORB[480.00,486.55] vol=1.7x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 13:30:00 | 491.68 | 487.44 | 0.00 | T1 1.5R @ 491.68 |
| Target hit | 2025-10-16 15:20:00 | 490.15 | 488.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:15:00 | 496.95 | 498.79 | 0.00 | ORB-short ORB[500.10,505.00] vol=4.5x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 11:20:00 | 494.37 | 498.65 | 0.00 | T1 1.5R @ 494.37 |
| Target hit | 2025-10-20 15:20:00 | 496.30 | 496.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:35:00 | 495.30 | 492.96 | 0.00 | ORB-long ORB[487.65,493.90] vol=2.2x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 09:55:00 | 497.95 | 493.78 | 0.00 | T1 1.5R @ 497.95 |
| Stop hit — per-position SL triggered | 2025-10-24 10:00:00 | 495.30 | 494.09 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 494.00 | 490.78 | 0.00 | ORB-long ORB[486.90,491.95] vol=2.4x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-10-28 09:35:00 | 492.03 | 491.52 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:50:00 | 495.85 | 491.72 | 0.00 | ORB-long ORB[489.00,495.00] vol=1.7x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:25:00 | 497.83 | 492.99 | 0.00 | T1 1.5R @ 497.83 |
| Stop hit — per-position SL triggered | 2025-11-03 11:30:00 | 495.85 | 493.53 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:55:00 | 488.30 | 485.97 | 0.00 | ORB-long ORB[484.35,486.90] vol=1.8x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:00:00 | 490.27 | 486.72 | 0.00 | T1 1.5R @ 490.27 |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 488.30 | 489.69 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:55:00 | 495.00 | 492.38 | 0.00 | ORB-long ORB[488.05,492.00] vol=2.2x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:05:00 | 496.88 | 492.89 | 0.00 | T1 1.5R @ 496.88 |
| Stop hit — per-position SL triggered | 2025-11-10 12:30:00 | 495.00 | 494.06 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:45:00 | 492.80 | 488.86 | 0.00 | ORB-long ORB[486.75,492.60] vol=2.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 11:00:00 | 494.85 | 490.19 | 0.00 | T1 1.5R @ 494.85 |
| Stop hit — per-position SL triggered | 2025-11-12 11:10:00 | 492.80 | 490.95 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:30:00 | 500.50 | 498.86 | 0.00 | ORB-long ORB[495.80,500.00] vol=2.6x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:45:00 | 502.98 | 500.10 | 0.00 | T1 1.5R @ 502.98 |
| Target hit | 2025-11-13 12:10:00 | 500.95 | 502.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:55:00 | 499.30 | 501.52 | 0.00 | ORB-short ORB[500.15,504.30] vol=2.4x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:05:00 | 497.16 | 500.70 | 0.00 | T1 1.5R @ 497.16 |
| Target hit | 2025-11-21 15:20:00 | 492.05 | 494.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-11-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 09:55:00 | 484.50 | 486.88 | 0.00 | ORB-short ORB[487.05,493.95] vol=4.2x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 482.10 | 485.46 | 0.00 | T1 1.5R @ 482.10 |
| Stop hit — per-position SL triggered | 2025-11-24 11:35:00 | 484.50 | 485.18 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-26 10:30:00 | 488.70 | 493.13 | 0.00 | ORB-short ORB[492.50,496.95] vol=2.4x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 10:40:00 | 486.36 | 490.04 | 0.00 | T1 1.5R @ 486.36 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 488.70 | 489.06 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:45:00 | 490.75 | 490.42 | 0.00 | ORB-long ORB[485.95,490.00] vol=18.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-12-01 10:50:00 | 489.42 | 490.30 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:10:00 | 462.15 | 461.52 | 0.00 | ORB-long ORB[456.80,461.80] vol=15.1x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:15:00 | 465.33 | 461.56 | 0.00 | T1 1.5R @ 465.33 |
| Target hit | 2025-12-09 15:20:00 | 470.45 | 464.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:15:00 | 473.00 | 471.68 | 0.00 | ORB-long ORB[467.30,471.85] vol=2.0x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-12-10 10:50:00 | 471.44 | 471.69 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 464.50 | 465.85 | 0.00 | ORB-short ORB[464.85,468.65] vol=1.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-12-12 10:00:00 | 465.43 | 465.39 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:35:00 | 467.25 | 464.95 | 0.00 | ORB-long ORB[462.35,466.50] vol=2.1x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-12-16 10:45:00 | 465.66 | 465.48 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:05:00 | 461.50 | 463.05 | 0.00 | ORB-short ORB[462.15,468.05] vol=2.0x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 12:40:00 | 459.97 | 462.27 | 0.00 | T1 1.5R @ 459.97 |
| Stop hit — per-position SL triggered | 2025-12-22 15:00:00 | 461.50 | 461.28 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 439.65 | 442.08 | 0.00 | ORB-short ORB[441.00,446.00] vol=5.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-12-29 11:30:00 | 440.69 | 441.77 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:30:00 | 446.15 | 444.77 | 0.00 | ORB-long ORB[441.70,444.70] vol=2.3x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-12-30 09:40:00 | 444.76 | 445.16 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:55:00 | 450.20 | 448.39 | 0.00 | ORB-long ORB[446.35,448.55] vol=2.4x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-12-31 11:10:00 | 449.09 | 448.44 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:10:00 | 452.50 | 453.99 | 0.00 | ORB-short ORB[455.15,458.80] vol=2.4x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:45:00 | 450.87 | 453.33 | 0.00 | T1 1.5R @ 450.87 |
| Stop hit — per-position SL triggered | 2026-01-06 10:50:00 | 452.50 | 453.21 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:25:00 | 455.90 | 450.59 | 0.00 | ORB-long ORB[442.85,449.20] vol=1.8x ATR=1.84 |
| Stop hit — per-position SL triggered | 2026-01-09 11:50:00 | 454.06 | 454.53 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 10:10:00 | 439.50 | 435.97 | 0.00 | ORB-long ORB[432.45,438.70] vol=2.3x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:25:00 | 442.00 | 437.26 | 0.00 | T1 1.5R @ 442.00 |
| Stop hit — per-position SL triggered | 2026-01-20 10:35:00 | 439.50 | 437.59 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:30:00 | 428.75 | 431.48 | 0.00 | ORB-short ORB[433.45,436.35] vol=2.9x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:05:00 | 427.17 | 430.77 | 0.00 | T1 1.5R @ 427.17 |
| Stop hit — per-position SL triggered | 2026-01-23 12:25:00 | 428.75 | 427.80 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:40:00 | 459.65 | 464.00 | 0.00 | ORB-short ORB[463.30,468.60] vol=1.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-02-05 10:10:00 | 461.06 | 462.64 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 467.10 | 464.14 | 0.00 | ORB-long ORB[460.65,465.75] vol=5.6x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 469.14 | 465.61 | 0.00 | T1 1.5R @ 469.14 |
| Stop hit — per-position SL triggered | 2026-02-10 13:30:00 | 467.10 | 467.74 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:15:00 | 471.25 | 469.61 | 0.00 | ORB-long ORB[467.10,471.05] vol=2.0x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:10:00 | 473.25 | 470.74 | 0.00 | T1 1.5R @ 473.25 |
| Target hit | 2026-02-11 13:50:00 | 480.95 | 480.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 481.50 | 478.54 | 0.00 | ORB-long ORB[475.00,479.75] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-02-13 09:45:00 | 479.71 | 478.60 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 469.45 | 468.48 | 0.00 | ORB-long ORB[462.20,469.10] vol=1.7x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-02-16 11:45:00 | 468.29 | 468.52 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 479.15 | 480.99 | 0.00 | ORB-short ORB[479.50,485.60] vol=1.6x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 14:00:00 | 476.95 | 480.07 | 0.00 | T1 1.5R @ 476.95 |
| Target hit | 2026-02-18 15:20:00 | 472.45 | 478.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 453.15 | 455.60 | 0.00 | ORB-short ORB[454.10,459.50] vol=6.6x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:20:00 | 451.38 | 455.13 | 0.00 | T1 1.5R @ 451.38 |
| Stop hit — per-position SL triggered | 2026-02-24 11:25:00 | 453.15 | 455.12 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 454.55 | 452.58 | 0.00 | ORB-long ORB[446.15,451.80] vol=3.8x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 13:05:00 | 456.62 | 453.82 | 0.00 | T1 1.5R @ 456.62 |
| Stop hit — per-position SL triggered | 2026-03-10 13:50:00 | 454.55 | 454.11 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 455.60 | 451.74 | 0.00 | ORB-long ORB[447.05,452.20] vol=4.2x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:00:00 | 457.35 | 452.55 | 0.00 | T1 1.5R @ 457.35 |
| Stop hit — per-position SL triggered | 2026-03-11 11:45:00 | 455.60 | 456.09 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 09:40:00 | 461.50 | 459.83 | 0.00 | ORB-long ORB[456.60,461.20] vol=1.5x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:45:00 | 463.46 | 460.34 | 0.00 | T1 1.5R @ 463.46 |
| Target hit | 2026-03-19 10:25:00 | 463.30 | 463.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 67 — BUY (started 2026-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 10:30:00 | 460.35 | 456.00 | 0.00 | ORB-long ORB[451.80,457.10] vol=2.3x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-03-27 10:40:00 | 458.53 | 457.00 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-04-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:45:00 | 451.35 | 453.18 | 0.00 | ORB-short ORB[452.10,458.05] vol=1.5x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-04-01 11:45:00 | 453.39 | 452.65 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-04-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:55:00 | 471.60 | 469.21 | 0.00 | ORB-long ORB[465.55,469.60] vol=4.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-04-08 10:40:00 | 469.71 | 470.72 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 466.25 | 470.50 | 0.00 | ORB-short ORB[468.00,475.00] vol=1.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-04-09 10:00:00 | 467.54 | 468.90 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 481.70 | 479.26 | 0.00 | ORB-long ORB[475.00,480.00] vol=1.8x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:40:00 | 484.53 | 482.48 | 0.00 | T1 1.5R @ 484.53 |
| Target hit | 2026-04-15 13:05:00 | 484.40 | 486.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 487.85 | 485.48 | 0.00 | ORB-long ORB[480.05,486.35] vol=3.4x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:10:00 | 490.61 | 486.66 | 0.00 | T1 1.5R @ 490.61 |
| Target hit | 2026-04-16 15:20:00 | 496.50 | 492.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2026-04-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:50:00 | 517.00 | 512.99 | 0.00 | ORB-long ORB[505.55,511.70] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 515.42 | 514.03 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 519.25 | 517.17 | 0.00 | ORB-long ORB[513.90,517.95] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-04-28 11:25:00 | 517.74 | 518.22 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 523.90 | 521.94 | 0.00 | ORB-long ORB[519.00,523.20] vol=1.8x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-05-06 09:45:00 | 522.16 | 521.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-21 10:35:00 | 443.75 | 2025-05-21 10:55:00 | 441.93 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-05-23 10:10:00 | 447.60 | 2025-05-23 11:00:00 | 450.01 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-23 10:10:00 | 447.60 | 2025-05-23 15:20:00 | 462.30 | TARGET_HIT | 0.50 | 3.28% |
| BUY | retest1 | 2025-05-28 09:40:00 | 470.65 | 2025-05-28 10:00:00 | 468.36 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-06-02 10:35:00 | 473.90 | 2025-06-02 11:40:00 | 475.75 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-06-04 11:05:00 | 472.35 | 2025-06-04 12:00:00 | 470.63 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-06-04 11:05:00 | 472.35 | 2025-06-04 15:20:00 | 466.95 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-06-16 10:10:00 | 438.95 | 2025-06-16 10:15:00 | 440.34 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-19 10:15:00 | 430.60 | 2025-06-19 10:40:00 | 428.37 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-19 10:15:00 | 430.60 | 2025-06-19 15:05:00 | 430.15 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-06-25 10:15:00 | 434.80 | 2025-06-25 10:20:00 | 438.68 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2025-06-25 10:15:00 | 434.80 | 2025-06-25 10:25:00 | 434.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:15:00 | 414.00 | 2025-07-01 10:20:00 | 415.48 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-03 10:50:00 | 416.25 | 2025-07-03 12:10:00 | 417.52 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-08 09:45:00 | 426.90 | 2025-07-08 09:50:00 | 425.63 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-09 11:00:00 | 430.40 | 2025-07-09 11:20:00 | 432.11 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-07-09 11:00:00 | 430.40 | 2025-07-09 15:10:00 | 431.15 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-07-10 10:40:00 | 428.70 | 2025-07-10 12:05:00 | 426.81 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-07-10 10:40:00 | 428.70 | 2025-07-10 15:20:00 | 427.10 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-11 10:25:00 | 423.25 | 2025-07-11 10:40:00 | 421.53 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-11 10:25:00 | 423.25 | 2025-07-11 13:05:00 | 421.20 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-16 09:40:00 | 435.20 | 2025-07-16 09:55:00 | 437.80 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-07-16 09:40:00 | 435.20 | 2025-07-16 11:40:00 | 440.05 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2025-07-22 10:05:00 | 439.70 | 2025-07-22 10:15:00 | 440.76 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-29 10:45:00 | 420.00 | 2025-07-29 11:00:00 | 421.38 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-07 11:00:00 | 433.50 | 2025-08-07 11:40:00 | 432.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-11 09:30:00 | 440.40 | 2025-08-11 09:40:00 | 438.64 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-08-14 10:05:00 | 441.40 | 2025-08-14 10:10:00 | 443.05 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-19 11:00:00 | 443.00 | 2025-08-19 11:15:00 | 440.98 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-08-19 11:00:00 | 443.00 | 2025-08-19 11:25:00 | 443.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 10:15:00 | 443.50 | 2025-08-22 12:00:00 | 441.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-08-25 10:30:00 | 438.80 | 2025-08-25 12:00:00 | 440.41 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-03 10:05:00 | 449.05 | 2025-09-03 10:10:00 | 447.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-24 09:30:00 | 454.10 | 2025-09-24 09:35:00 | 452.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-25 10:40:00 | 457.70 | 2025-09-25 11:50:00 | 456.37 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-26 09:30:00 | 447.60 | 2025-09-26 09:40:00 | 449.40 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-07 09:35:00 | 455.30 | 2025-10-07 10:00:00 | 456.71 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-08 10:15:00 | 479.05 | 2025-10-08 10:30:00 | 477.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-10-10 09:55:00 | 475.65 | 2025-10-10 10:05:00 | 477.53 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-14 09:35:00 | 472.95 | 2025-10-14 09:45:00 | 474.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-16 09:40:00 | 487.40 | 2025-10-16 13:30:00 | 491.68 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2025-10-16 09:40:00 | 487.40 | 2025-10-16 15:20:00 | 490.15 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-10-20 11:15:00 | 496.95 | 2025-10-20 11:20:00 | 494.37 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-10-20 11:15:00 | 496.95 | 2025-10-20 15:20:00 | 496.30 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2025-10-24 09:35:00 | 495.30 | 2025-10-24 09:55:00 | 497.95 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-24 09:35:00 | 495.30 | 2025-10-24 10:00:00 | 495.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 09:30:00 | 494.00 | 2025-10-28 09:35:00 | 492.03 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-11-03 10:50:00 | 495.85 | 2025-11-03 11:25:00 | 497.83 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-11-03 10:50:00 | 495.85 | 2025-11-03 11:30:00 | 495.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-04 09:55:00 | 488.30 | 2025-11-04 10:00:00 | 490.27 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-11-04 09:55:00 | 488.30 | 2025-11-04 10:15:00 | 488.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 10:55:00 | 495.00 | 2025-11-10 11:05:00 | 496.88 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-11-10 10:55:00 | 495.00 | 2025-11-10 12:30:00 | 495.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 10:45:00 | 492.80 | 2025-11-12 11:00:00 | 494.85 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-11-12 10:45:00 | 492.80 | 2025-11-12 11:10:00 | 492.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 09:30:00 | 500.50 | 2025-11-13 09:45:00 | 502.98 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-13 09:30:00 | 500.50 | 2025-11-13 12:10:00 | 500.95 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2025-11-21 09:55:00 | 499.30 | 2025-11-21 10:05:00 | 497.16 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-21 09:55:00 | 499.30 | 2025-11-21 15:20:00 | 492.05 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2025-11-24 09:55:00 | 484.50 | 2025-11-24 11:15:00 | 482.10 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-24 09:55:00 | 484.50 | 2025-11-24 11:35:00 | 484.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-26 10:30:00 | 488.70 | 2025-11-26 10:40:00 | 486.36 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-11-26 10:30:00 | 488.70 | 2025-11-26 11:15:00 | 488.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 10:45:00 | 490.75 | 2025-12-01 10:50:00 | 489.42 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-09 10:10:00 | 462.15 | 2025-12-09 10:15:00 | 465.33 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-12-09 10:10:00 | 462.15 | 2025-12-09 15:20:00 | 470.45 | TARGET_HIT | 0.50 | 1.80% |
| BUY | retest1 | 2025-12-10 10:15:00 | 473.00 | 2025-12-10 10:50:00 | 471.44 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-12 09:30:00 | 464.50 | 2025-12-12 10:00:00 | 465.43 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-16 10:35:00 | 467.25 | 2025-12-16 10:45:00 | 465.66 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-22 11:05:00 | 461.50 | 2025-12-22 12:40:00 | 459.97 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-22 11:05:00 | 461.50 | 2025-12-22 15:00:00 | 461.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 11:05:00 | 439.65 | 2025-12-29 11:30:00 | 440.69 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-30 09:30:00 | 446.15 | 2025-12-30 09:40:00 | 444.76 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-31 10:55:00 | 450.20 | 2025-12-31 11:10:00 | 449.09 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-06 10:10:00 | 452.50 | 2026-01-06 10:45:00 | 450.87 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-06 10:10:00 | 452.50 | 2026-01-06 10:50:00 | 452.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-09 10:25:00 | 455.90 | 2026-01-09 11:50:00 | 454.06 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-20 10:10:00 | 439.50 | 2026-01-20 10:25:00 | 442.00 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-01-20 10:10:00 | 439.50 | 2026-01-20 10:35:00 | 439.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-23 10:30:00 | 428.75 | 2026-01-23 11:05:00 | 427.17 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-23 10:30:00 | 428.75 | 2026-01-23 12:25:00 | 428.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 09:40:00 | 459.65 | 2026-02-05 10:10:00 | 461.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-10 10:35:00 | 467.10 | 2026-02-10 10:40:00 | 469.14 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-10 10:35:00 | 467.10 | 2026-02-10 13:30:00 | 467.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:15:00 | 471.25 | 2026-02-11 12:10:00 | 473.25 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-11 10:15:00 | 471.25 | 2026-02-11 13:50:00 | 480.95 | TARGET_HIT | 0.50 | 2.06% |
| BUY | retest1 | 2026-02-13 09:40:00 | 481.50 | 2026-02-13 09:45:00 | 479.71 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-16 10:45:00 | 469.45 | 2026-02-16 11:45:00 | 468.29 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-18 10:45:00 | 479.15 | 2026-02-18 14:00:00 | 476.95 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-18 10:45:00 | 479.15 | 2026-02-18 15:20:00 | 472.45 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2026-02-24 11:00:00 | 453.15 | 2026-02-24 11:20:00 | 451.38 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-24 11:00:00 | 453.15 | 2026-02-24 11:25:00 | 453.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:50:00 | 454.55 | 2026-03-10 13:05:00 | 456.62 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-10 10:50:00 | 454.55 | 2026-03-10 13:50:00 | 454.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 10:55:00 | 455.60 | 2026-03-11 11:00:00 | 457.35 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-03-11 10:55:00 | 455.60 | 2026-03-11 11:45:00 | 455.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-19 09:40:00 | 461.50 | 2026-03-19 09:45:00 | 463.46 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-03-19 09:40:00 | 461.50 | 2026-03-19 10:25:00 | 463.30 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-27 10:30:00 | 460.35 | 2026-03-27 10:40:00 | 458.53 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-01 10:45:00 | 451.35 | 2026-04-01 11:45:00 | 453.39 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-08 09:55:00 | 471.60 | 2026-04-08 10:40:00 | 469.71 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-09 09:45:00 | 466.25 | 2026-04-09 10:00:00 | 467.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-15 09:35:00 | 481.70 | 2026-04-15 09:40:00 | 484.53 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-15 09:35:00 | 481.70 | 2026-04-15 13:05:00 | 484.40 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-16 09:55:00 | 487.85 | 2026-04-16 10:10:00 | 490.61 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-16 09:55:00 | 487.85 | 2026-04-16 15:20:00 | 496.50 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2026-04-23 10:50:00 | 517.00 | 2026-04-23 11:05:00 | 515.42 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-28 09:55:00 | 519.25 | 2026-04-28 11:25:00 | 517.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-06 09:40:00 | 523.90 | 2026-05-06 09:45:00 | 522.16 | STOP_HIT | 1.00 | -0.33% |
