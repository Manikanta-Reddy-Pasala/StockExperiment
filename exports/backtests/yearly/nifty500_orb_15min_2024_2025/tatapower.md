# Tata Power Co. Ltd. (TATAPOWER)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 435.50
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
| ENTRY1 | 82 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 9 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 106 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 73
- **Target hits / Stop hits / Partials:** 9 / 73 / 24
- **Avg / median % per leg:** 0.08% / -0.23%
- **Sum % (uncompounded):** 8.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 67 | 18 | 26.9% | 3 | 49 | 15 | 0.02% | 1.4% |
| BUY @ 2nd Alert (retest1) | 67 | 18 | 26.9% | 3 | 49 | 15 | 0.02% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 15 | 38.5% | 6 | 24 | 9 | 0.19% | 7.4% |
| SELL @ 2nd Alert (retest1) | 39 | 15 | 38.5% | 6 | 24 | 9 | 0.19% | 7.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 106 | 33 | 31.1% | 9 | 73 | 24 | 0.08% | 8.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:50:00 | 441.00 | 439.27 | 0.00 | ORB-long ORB[436.50,437.90] vol=3.0x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-05-18 11:30:00 | 439.97 | 440.00 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 438.80 | 442.65 | 0.00 | ORB-short ORB[441.50,444.95] vol=2.1x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 440.37 | 441.63 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 450.80 | 449.29 | 0.00 | ORB-long ORB[447.05,450.50] vol=2.1x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 449.61 | 449.57 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:40:00 | 449.75 | 447.88 | 0.00 | ORB-long ORB[447.00,449.50] vol=2.2x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-05-24 10:45:00 | 448.69 | 447.95 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 451.80 | 449.74 | 0.00 | ORB-long ORB[447.25,450.75] vol=2.8x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-05-27 09:40:00 | 450.14 | 449.87 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 444.25 | 446.26 | 0.00 | ORB-short ORB[446.55,449.40] vol=2.1x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:05:00 | 442.30 | 444.94 | 0.00 | T1 1.5R @ 442.30 |
| Target hit | 2024-05-28 15:20:00 | 437.15 | 439.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 433.60 | 431.46 | 0.00 | ORB-long ORB[427.00,433.30] vol=1.6x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-06-07 09:45:00 | 431.98 | 431.69 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:40:00 | 457.70 | 455.56 | 0.00 | ORB-long ORB[453.75,457.00] vol=4.1x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-06-13 09:45:00 | 456.28 | 455.64 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 439.95 | 442.38 | 0.00 | ORB-short ORB[440.45,444.70] vol=1.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-06-21 11:00:00 | 441.16 | 442.24 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 432.00 | 435.48 | 0.00 | ORB-short ORB[434.50,437.95] vol=1.9x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 432.98 | 435.41 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:40:00 | 430.35 | 432.16 | 0.00 | ORB-short ORB[431.80,434.30] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-06-27 11:30:00 | 431.52 | 431.72 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:35:00 | 439.05 | 436.62 | 0.00 | ORB-long ORB[433.25,437.20] vol=3.0x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-07-04 09:40:00 | 437.84 | 437.18 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:15:00 | 438.50 | 436.35 | 0.00 | ORB-long ORB[434.50,437.10] vol=1.9x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 11:30:00 | 439.72 | 436.70 | 0.00 | T1 1.5R @ 439.72 |
| Stop hit — per-position SL triggered | 2024-07-05 12:45:00 | 438.50 | 437.72 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 438.80 | 436.73 | 0.00 | ORB-long ORB[434.70,437.80] vol=1.6x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:35:00 | 440.49 | 438.20 | 0.00 | T1 1.5R @ 440.49 |
| Stop hit — per-position SL triggered | 2024-07-09 09:55:00 | 438.80 | 438.91 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:40:00 | 441.70 | 439.50 | 0.00 | ORB-long ORB[437.20,440.40] vol=3.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-07-10 09:45:00 | 440.64 | 439.73 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:40:00 | 442.00 | 438.94 | 0.00 | ORB-long ORB[436.10,439.95] vol=2.8x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 09:45:00 | 444.16 | 440.77 | 0.00 | T1 1.5R @ 444.16 |
| Stop hit — per-position SL triggered | 2024-07-11 09:50:00 | 442.00 | 440.93 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 11:15:00 | 437.25 | 434.06 | 0.00 | ORB-long ORB[431.00,436.55] vol=1.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-07-15 12:05:00 | 436.18 | 434.64 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 443.00 | 441.14 | 0.00 | ORB-long ORB[438.90,440.95] vol=3.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 09:40:00 | 444.82 | 442.31 | 0.00 | T1 1.5R @ 444.82 |
| Stop hit — per-position SL triggered | 2024-07-16 09:50:00 | 443.00 | 442.59 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 11:15:00 | 444.50 | 446.17 | 0.00 | ORB-short ORB[446.20,450.00] vol=2.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2024-07-29 11:30:00 | 445.80 | 446.07 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:15:00 | 447.80 | 444.75 | 0.00 | ORB-long ORB[440.00,446.50] vol=1.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 446.32 | 445.91 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:05:00 | 459.80 | 457.38 | 0.00 | ORB-long ORB[454.05,457.85] vol=2.4x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:20:00 | 462.33 | 458.39 | 0.00 | T1 1.5R @ 462.33 |
| Stop hit — per-position SL triggered | 2024-08-01 10:35:00 | 459.80 | 458.78 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:10:00 | 423.80 | 425.38 | 0.00 | ORB-short ORB[424.00,427.30] vol=2.2x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-08-21 11:25:00 | 424.67 | 425.35 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 11:10:00 | 432.30 | 431.18 | 0.00 | ORB-long ORB[427.15,431.50] vol=5.2x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-08-29 11:30:00 | 431.16 | 431.21 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:00:00 | 435.15 | 433.74 | 0.00 | ORB-long ORB[431.35,434.30] vol=3.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-08-30 10:15:00 | 433.82 | 433.81 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 425.65 | 426.67 | 0.00 | ORB-short ORB[426.05,429.70] vol=4.4x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-09-04 10:00:00 | 426.76 | 426.21 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:40:00 | 426.85 | 423.65 | 0.00 | ORB-long ORB[421.85,425.80] vol=1.9x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:55:00 | 429.08 | 425.26 | 0.00 | T1 1.5R @ 429.08 |
| Target hit | 2024-09-10 15:20:00 | 446.05 | 437.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-09-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:40:00 | 448.45 | 444.58 | 0.00 | ORB-long ORB[441.00,446.45] vol=1.6x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-09-11 09:45:00 | 446.25 | 445.04 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 449.25 | 446.42 | 0.00 | ORB-long ORB[442.80,448.20] vol=3.7x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-09-13 09:35:00 | 447.58 | 446.60 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 447.70 | 445.90 | 0.00 | ORB-long ORB[441.00,447.65] vol=2.2x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 09:35:00 | 449.73 | 447.58 | 0.00 | T1 1.5R @ 449.73 |
| Stop hit — per-position SL triggered | 2024-09-16 09:45:00 | 447.70 | 447.89 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:10:00 | 440.70 | 442.81 | 0.00 | ORB-short ORB[442.00,446.15] vol=1.8x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-09-17 10:35:00 | 441.85 | 442.46 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 448.90 | 445.97 | 0.00 | ORB-long ORB[442.40,446.60] vol=4.0x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 447.22 | 447.14 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:05:00 | 473.20 | 470.25 | 0.00 | ORB-long ORB[467.65,470.85] vol=1.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:20:00 | 476.17 | 471.20 | 0.00 | T1 1.5R @ 476.17 |
| Stop hit — per-position SL triggered | 2024-09-26 11:40:00 | 473.20 | 472.84 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 491.25 | 484.84 | 0.00 | ORB-long ORB[478.60,485.90] vol=2.3x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-09-27 10:50:00 | 488.78 | 486.88 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:50:00 | 489.95 | 488.02 | 0.00 | ORB-long ORB[485.35,489.80] vol=3.2x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-10-01 09:55:00 | 488.07 | 488.09 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:10:00 | 459.35 | 461.94 | 0.00 | ORB-short ORB[460.30,464.55] vol=1.7x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-10-14 11:40:00 | 460.51 | 461.71 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 470.00 | 467.51 | 0.00 | ORB-long ORB[461.55,468.45] vol=4.5x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:35:00 | 472.42 | 468.59 | 0.00 | T1 1.5R @ 472.42 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 470.00 | 469.43 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:35:00 | 468.50 | 465.86 | 0.00 | ORB-long ORB[462.10,466.85] vol=2.2x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-10-16 09:40:00 | 467.01 | 466.10 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 450.50 | 452.77 | 0.00 | ORB-short ORB[452.00,458.00] vol=1.9x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-10-21 09:40:00 | 451.89 | 452.62 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:30:00 | 423.80 | 425.60 | 0.00 | ORB-short ORB[424.10,428.55] vol=1.9x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:40:00 | 421.67 | 424.72 | 0.00 | T1 1.5R @ 421.67 |
| Target hit | 2024-10-29 12:00:00 | 420.80 | 420.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2024-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:30:00 | 443.40 | 440.64 | 0.00 | ORB-long ORB[436.50,442.20] vol=2.9x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-11-06 09:40:00 | 441.81 | 441.16 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 11:00:00 | 444.60 | 449.94 | 0.00 | ORB-short ORB[446.50,452.20] vol=1.6x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-11-07 15:20:00 | 445.40 | 447.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-11-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:00:00 | 434.75 | 432.63 | 0.00 | ORB-long ORB[428.05,433.70] vol=1.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:15:00 | 437.05 | 433.37 | 0.00 | T1 1.5R @ 437.05 |
| Stop hit — per-position SL triggered | 2024-11-11 12:05:00 | 434.75 | 433.82 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:40:00 | 428.30 | 431.88 | 0.00 | ORB-short ORB[432.05,436.35] vol=2.3x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 10:45:00 | 425.98 | 429.45 | 0.00 | T1 1.5R @ 425.98 |
| Target hit | 2024-11-12 15:20:00 | 414.40 | 421.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-11-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:35:00 | 399.30 | 403.41 | 0.00 | ORB-short ORB[402.50,408.00] vol=2.2x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 400.97 | 402.20 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 09:35:00 | 411.00 | 413.96 | 0.00 | ORB-short ORB[412.10,416.90] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-11-22 11:50:00 | 412.71 | 412.45 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:30:00 | 415.10 | 412.32 | 0.00 | ORB-long ORB[409.25,412.35] vol=1.7x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-11-27 10:50:00 | 413.94 | 412.60 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:50:00 | 415.95 | 418.40 | 0.00 | ORB-short ORB[416.90,420.50] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-11-28 11:20:00 | 417.16 | 417.73 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:45:00 | 424.30 | 421.93 | 0.00 | ORB-long ORB[417.60,423.40] vol=2.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:00:00 | 426.24 | 423.08 | 0.00 | T1 1.5R @ 426.24 |
| Stop hit — per-position SL triggered | 2024-12-03 11:30:00 | 424.30 | 424.66 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:45:00 | 422.00 | 424.90 | 0.00 | ORB-short ORB[424.15,427.40] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-12-05 09:50:00 | 423.11 | 424.72 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:45:00 | 430.25 | 432.85 | 0.00 | ORB-short ORB[432.35,436.75] vol=1.7x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 431.19 | 432.73 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 09:55:00 | 415.75 | 417.99 | 0.00 | ORB-short ORB[416.20,421.00] vol=1.6x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-12-18 10:20:00 | 417.17 | 417.64 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:55:00 | 403.60 | 400.83 | 0.00 | ORB-long ORB[399.35,402.00] vol=1.6x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-12-24 11:30:00 | 402.44 | 402.40 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:30:00 | 399.00 | 399.58 | 0.00 | ORB-short ORB[399.10,401.80] vol=2.2x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-12-26 10:55:00 | 399.94 | 399.50 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:20:00 | 405.85 | 406.89 | 0.00 | ORB-short ORB[405.95,408.00] vol=1.5x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:40:00 | 404.39 | 406.44 | 0.00 | T1 1.5R @ 404.39 |
| Target hit | 2024-12-27 15:20:00 | 399.45 | 402.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2024-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 11:00:00 | 396.70 | 398.48 | 0.00 | ORB-short ORB[397.75,400.30] vol=2.0x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-12-30 11:20:00 | 397.66 | 398.40 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:10:00 | 389.85 | 390.97 | 0.00 | ORB-short ORB[390.10,394.00] vol=1.7x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:35:00 | 388.49 | 390.24 | 0.00 | T1 1.5R @ 388.49 |
| Stop hit — per-position SL triggered | 2025-01-02 11:55:00 | 389.85 | 389.75 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 369.75 | 371.88 | 0.00 | ORB-short ORB[370.65,376.00] vol=2.9x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:30:00 | 368.26 | 371.17 | 0.00 | T1 1.5R @ 368.26 |
| Target hit | 2025-01-09 15:20:00 | 367.20 | 368.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2025-01-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 09:45:00 | 361.45 | 358.45 | 0.00 | ORB-long ORB[356.10,360.50] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:15:00 | 363.68 | 360.27 | 0.00 | T1 1.5R @ 363.68 |
| Target hit | 2025-01-15 13:30:00 | 365.25 | 365.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2025-01-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:30:00 | 373.20 | 370.42 | 0.00 | ORB-long ORB[368.10,373.05] vol=1.6x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-01-16 10:45:00 | 371.90 | 370.73 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:00:00 | 372.50 | 369.04 | 0.00 | ORB-long ORB[365.30,368.60] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-01-17 10:40:00 | 371.50 | 370.43 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:05:00 | 369.80 | 373.39 | 0.00 | ORB-short ORB[372.75,375.70] vol=1.8x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:20:00 | 368.18 | 372.44 | 0.00 | T1 1.5R @ 368.18 |
| Stop hit — per-position SL triggered | 2025-01-21 11:40:00 | 369.80 | 370.62 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:30:00 | 363.00 | 359.30 | 0.00 | ORB-long ORB[355.10,358.70] vol=1.8x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-01-23 11:05:00 | 361.69 | 360.23 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:15:00 | 354.25 | 353.53 | 0.00 | ORB-long ORB[349.10,353.15] vol=2.0x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-01-30 10:35:00 | 353.10 | 353.67 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:05:00 | 359.35 | 354.77 | 0.00 | ORB-long ORB[350.10,354.25] vol=2.0x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 358.17 | 355.35 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:45:00 | 372.50 | 368.49 | 0.00 | ORB-long ORB[364.25,368.35] vol=1.9x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-02-01 09:50:00 | 370.94 | 368.94 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:15:00 | 367.00 | 369.41 | 0.00 | ORB-short ORB[367.55,371.85] vol=2.0x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-02-06 11:20:00 | 367.96 | 369.37 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:00:00 | 370.95 | 367.53 | 0.00 | ORB-long ORB[364.15,369.70] vol=2.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-02-07 12:05:00 | 369.42 | 368.58 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 363.05 | 365.11 | 0.00 | ORB-short ORB[364.20,368.95] vol=2.0x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:40:00 | 361.17 | 364.39 | 0.00 | T1 1.5R @ 361.17 |
| Stop hit — per-position SL triggered | 2025-02-10 09:50:00 | 363.05 | 364.04 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 354.55 | 352.90 | 0.00 | ORB-long ORB[350.30,354.15] vol=1.6x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:40:00 | 356.80 | 353.75 | 0.00 | T1 1.5R @ 356.80 |
| Stop hit — per-position SL triggered | 2025-02-13 10:20:00 | 354.55 | 354.28 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 349.90 | 347.82 | 0.00 | ORB-long ORB[344.55,348.65] vol=2.0x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 348.70 | 348.09 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:25:00 | 350.25 | 348.66 | 0.00 | ORB-long ORB[344.55,348.65] vol=2.2x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 12:40:00 | 352.21 | 349.73 | 0.00 | T1 1.5R @ 352.21 |
| Stop hit — per-position SL triggered | 2025-03-05 13:20:00 | 350.25 | 349.95 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 359.05 | 356.26 | 0.00 | ORB-long ORB[352.10,356.70] vol=2.5x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-03-07 09:35:00 | 357.79 | 356.91 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 376.00 | 373.84 | 0.00 | ORB-long ORB[370.25,374.60] vol=2.4x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-03-21 09:45:00 | 375.00 | 374.28 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:15:00 | 380.85 | 378.47 | 0.00 | ORB-long ORB[375.50,380.35] vol=2.1x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-03-26 11:00:00 | 379.53 | 379.24 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:40:00 | 381.40 | 379.44 | 0.00 | ORB-long ORB[378.00,380.15] vol=1.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-04-16 09:50:00 | 380.29 | 379.62 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 11:00:00 | 379.95 | 380.80 | 0.00 | ORB-short ORB[380.00,384.35] vol=2.1x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-04-17 11:30:00 | 381.06 | 380.77 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 388.10 | 384.72 | 0.00 | ORB-long ORB[382.00,384.70] vol=3.2x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:45:00 | 389.88 | 386.19 | 0.00 | T1 1.5R @ 389.88 |
| Target hit | 2025-04-21 15:20:00 | 390.20 | 390.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2025-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:40:00 | 394.70 | 392.42 | 0.00 | ORB-long ORB[389.25,393.80] vol=1.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-04-22 09:50:00 | 393.55 | 392.69 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:00:00 | 397.30 | 396.37 | 0.00 | ORB-long ORB[391.90,395.40] vol=1.7x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-04-24 14:15:00 | 395.93 | 397.05 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:50:00 | 401.80 | 399.45 | 0.00 | ORB-long ORB[394.75,400.40] vol=1.6x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-04-29 10:05:00 | 400.45 | 400.09 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 11:10:00 | 380.50 | 384.86 | 0.00 | ORB-short ORB[385.20,389.80] vol=1.8x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 11:25:00 | 378.79 | 384.21 | 0.00 | T1 1.5R @ 378.79 |
| Target hit | 2025-05-06 15:20:00 | 374.95 | 379.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — BUY (started 2025-05-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:20:00 | 382.80 | 380.20 | 0.00 | ORB-long ORB[377.35,381.80] vol=2.2x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-05-08 10:35:00 | 381.54 | 380.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-18 09:50:00 | 441.00 | 2024-05-18 11:30:00 | 439.97 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-22 09:40:00 | 438.80 | 2024-05-22 09:55:00 | 440.37 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-23 09:30:00 | 450.80 | 2024-05-23 09:50:00 | 449.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-24 10:40:00 | 449.75 | 2024-05-24 10:45:00 | 448.69 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-05-27 09:35:00 | 451.80 | 2024-05-27 09:40:00 | 450.14 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-28 09:35:00 | 444.25 | 2024-05-28 10:05:00 | 442.30 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-28 09:35:00 | 444.25 | 2024-05-28 15:20:00 | 437.15 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2024-06-07 09:30:00 | 433.60 | 2024-06-07 09:45:00 | 431.98 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-13 09:40:00 | 457.70 | 2024-06-13 09:45:00 | 456.28 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-21 10:45:00 | 439.95 | 2024-06-21 11:00:00 | 441.16 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-25 11:15:00 | 432.00 | 2024-06-25 11:20:00 | 432.98 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-06-27 10:40:00 | 430.35 | 2024-06-27 11:30:00 | 431.52 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-04 09:35:00 | 439.05 | 2024-07-04 09:40:00 | 437.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-05 11:15:00 | 438.50 | 2024-07-05 11:30:00 | 439.72 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-07-05 11:15:00 | 438.50 | 2024-07-05 12:45:00 | 438.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 09:30:00 | 438.80 | 2024-07-09 09:35:00 | 440.49 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-09 09:30:00 | 438.80 | 2024-07-09 09:55:00 | 438.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-10 09:40:00 | 441.70 | 2024-07-10 09:45:00 | 440.64 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-11 09:40:00 | 442.00 | 2024-07-11 09:45:00 | 444.16 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-07-11 09:40:00 | 442.00 | 2024-07-11 09:50:00 | 442.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-15 11:15:00 | 437.25 | 2024-07-15 12:05:00 | 436.18 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-16 09:30:00 | 443.00 | 2024-07-16 09:40:00 | 444.82 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-16 09:30:00 | 443.00 | 2024-07-16 09:50:00 | 443.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-29 11:15:00 | 444.50 | 2024-07-29 11:30:00 | 445.80 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-30 10:15:00 | 447.80 | 2024-07-30 11:15:00 | 446.32 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-01 10:05:00 | 459.80 | 2024-08-01 10:20:00 | 462.33 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-01 10:05:00 | 459.80 | 2024-08-01 10:35:00 | 459.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-21 11:10:00 | 423.80 | 2024-08-21 11:25:00 | 424.67 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-08-29 11:10:00 | 432.30 | 2024-08-29 11:30:00 | 431.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-30 10:00:00 | 435.15 | 2024-08-30 10:15:00 | 433.82 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-04 09:30:00 | 425.65 | 2024-09-04 10:00:00 | 426.76 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-10 10:40:00 | 426.85 | 2024-09-10 10:55:00 | 429.08 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-10 10:40:00 | 426.85 | 2024-09-10 15:20:00 | 446.05 | TARGET_HIT | 0.50 | 4.50% |
| BUY | retest1 | 2024-09-11 09:40:00 | 448.45 | 2024-09-11 09:45:00 | 446.25 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-09-13 09:30:00 | 449.25 | 2024-09-13 09:35:00 | 447.58 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-16 09:30:00 | 447.70 | 2024-09-16 09:35:00 | 449.73 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-09-16 09:30:00 | 447.70 | 2024-09-16 09:45:00 | 447.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 10:10:00 | 440.70 | 2024-09-17 10:35:00 | 441.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-19 09:40:00 | 448.90 | 2024-09-19 09:50:00 | 447.22 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-26 10:05:00 | 473.20 | 2024-09-26 10:20:00 | 476.17 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-09-26 10:05:00 | 473.20 | 2024-09-26 11:40:00 | 473.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:15:00 | 491.25 | 2024-09-27 10:50:00 | 488.78 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-01 09:50:00 | 489.95 | 2024-10-01 09:55:00 | 488.07 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-14 11:10:00 | 459.35 | 2024-10-14 11:40:00 | 460.51 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-10-15 09:30:00 | 470.00 | 2024-10-15 09:35:00 | 472.42 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-10-15 09:30:00 | 470.00 | 2024-10-15 09:50:00 | 470.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 09:35:00 | 468.50 | 2024-10-16 09:40:00 | 467.01 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-21 09:35:00 | 450.50 | 2024-10-21 09:40:00 | 451.89 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-29 09:30:00 | 423.80 | 2024-10-29 09:40:00 | 421.67 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-29 09:30:00 | 423.80 | 2024-10-29 12:00:00 | 420.80 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2024-11-06 09:30:00 | 443.40 | 2024-11-06 09:40:00 | 441.81 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-07 11:00:00 | 444.60 | 2024-11-07 15:20:00 | 445.40 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-11-11 11:00:00 | 434.75 | 2024-11-11 11:15:00 | 437.05 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-11-11 11:00:00 | 434.75 | 2024-11-11 12:05:00 | 434.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 09:40:00 | 428.30 | 2024-11-12 10:45:00 | 425.98 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-11-12 09:40:00 | 428.30 | 2024-11-12 15:20:00 | 414.40 | TARGET_HIT | 0.50 | 3.25% |
| SELL | retest1 | 2024-11-18 09:35:00 | 399.30 | 2024-11-18 09:45:00 | 400.97 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-11-22 09:35:00 | 411.00 | 2024-11-22 11:50:00 | 412.71 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-11-27 10:30:00 | 415.10 | 2024-11-27 10:50:00 | 413.94 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-28 10:50:00 | 415.95 | 2024-11-28 11:20:00 | 417.16 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-03 09:45:00 | 424.30 | 2024-12-03 10:00:00 | 426.24 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-03 09:45:00 | 424.30 | 2024-12-03 11:30:00 | 424.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 09:45:00 | 422.00 | 2024-12-05 09:50:00 | 423.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-12 09:45:00 | 430.25 | 2024-12-12 09:50:00 | 431.19 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-18 09:55:00 | 415.75 | 2024-12-18 10:20:00 | 417.17 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-24 09:55:00 | 403.60 | 2024-12-24 11:30:00 | 402.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-26 10:30:00 | 399.00 | 2024-12-26 10:55:00 | 399.94 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-27 10:20:00 | 405.85 | 2024-12-27 10:40:00 | 404.39 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-27 10:20:00 | 405.85 | 2024-12-27 15:20:00 | 399.45 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2024-12-30 11:00:00 | 396.70 | 2024-12-30 11:20:00 | 397.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-02 10:10:00 | 389.85 | 2025-01-02 11:35:00 | 388.49 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-01-02 10:10:00 | 389.85 | 2025-01-02 11:55:00 | 389.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:45:00 | 369.75 | 2025-01-09 11:30:00 | 368.26 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-01-09 10:45:00 | 369.75 | 2025-01-09 15:20:00 | 367.20 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-15 09:45:00 | 361.45 | 2025-01-15 10:15:00 | 363.68 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-01-15 09:45:00 | 361.45 | 2025-01-15 13:30:00 | 365.25 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2025-01-16 10:30:00 | 373.20 | 2025-01-16 10:45:00 | 371.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-17 10:00:00 | 372.50 | 2025-01-17 10:40:00 | 371.50 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-21 10:05:00 | 369.80 | 2025-01-21 10:20:00 | 368.18 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-21 10:05:00 | 369.80 | 2025-01-21 11:40:00 | 369.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 10:30:00 | 363.00 | 2025-01-23 11:05:00 | 361.69 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-30 10:15:00 | 354.25 | 2025-01-30 10:35:00 | 353.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-31 11:05:00 | 359.35 | 2025-01-31 11:30:00 | 358.17 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-01 09:45:00 | 372.50 | 2025-02-01 09:50:00 | 370.94 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-02-06 11:15:00 | 367.00 | 2025-02-06 11:20:00 | 367.96 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-07 11:00:00 | 370.95 | 2025-02-07 12:05:00 | 369.42 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-02-10 09:30:00 | 363.05 | 2025-02-10 09:40:00 | 361.17 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-02-10 09:30:00 | 363.05 | 2025-02-10 09:50:00 | 363.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-13 09:30:00 | 354.55 | 2025-02-13 09:40:00 | 356.80 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-02-13 09:30:00 | 354.55 | 2025-02-13 10:20:00 | 354.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 09:35:00 | 349.90 | 2025-02-20 09:45:00 | 348.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-05 10:25:00 | 350.25 | 2025-03-05 12:40:00 | 352.21 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-03-05 10:25:00 | 350.25 | 2025-03-05 13:20:00 | 350.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 09:30:00 | 359.05 | 2025-03-07 09:35:00 | 357.79 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-21 09:30:00 | 376.00 | 2025-03-21 09:45:00 | 375.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-03-26 10:15:00 | 380.85 | 2025-03-26 11:00:00 | 379.53 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-16 09:40:00 | 381.40 | 2025-04-16 09:50:00 | 380.29 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-17 11:00:00 | 379.95 | 2025-04-17 11:30:00 | 381.06 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-21 09:35:00 | 388.10 | 2025-04-21 09:45:00 | 389.88 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-21 09:35:00 | 388.10 | 2025-04-21 15:20:00 | 390.20 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2025-04-22 09:40:00 | 394.70 | 2025-04-22 09:50:00 | 393.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-24 10:00:00 | 397.30 | 2025-04-24 14:15:00 | 395.93 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-29 09:50:00 | 401.80 | 2025-04-29 10:05:00 | 400.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-05-06 11:10:00 | 380.50 | 2025-05-06 11:25:00 | 378.79 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-06 11:10:00 | 380.50 | 2025-05-06 15:20:00 | 374.95 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2025-05-08 10:20:00 | 382.80 | 2025-05-08 10:35:00 | 381.54 | STOP_HIT | 1.00 | -0.33% |
