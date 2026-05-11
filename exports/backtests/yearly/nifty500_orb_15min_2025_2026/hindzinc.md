# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16963 bars)
- **Last close:** 634.75
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
| ENTRY1 | 67 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 8 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 87 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 59
- **Target hits / Stop hits / Partials:** 8 / 59 / 20
- **Avg / median % per leg:** -0.01% / -0.19%
- **Sum % (uncompounded):** -0.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 21 | 43.8% | 6 | 27 | 15 | 0.08% | 3.7% |
| BUY @ 2nd Alert (retest1) | 48 | 21 | 43.8% | 6 | 27 | 15 | 0.08% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 7 | 17.9% | 2 | 32 | 5 | -0.11% | -4.4% |
| SELL @ 2nd Alert (retest1) | 39 | 7 | 17.9% | 2 | 32 | 5 | -0.11% | -4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 87 | 28 | 32.2% | 8 | 59 | 20 | -0.01% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 436.70 | 434.65 | 0.00 | ORB-long ORB[429.80,436.00] vol=2.8x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 09:35:00 | 438.46 | 436.22 | 0.00 | T1 1.5R @ 438.46 |
| Stop hit — per-position SL triggered | 2025-05-14 09:50:00 | 436.70 | 436.82 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 443.60 | 441.26 | 0.00 | ORB-long ORB[438.25,443.10] vol=2.0x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 442.02 | 441.32 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:00:00 | 440.80 | 438.77 | 0.00 | ORB-long ORB[436.05,439.90] vol=2.1x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-05-21 11:30:00 | 439.53 | 438.99 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 11:00:00 | 443.50 | 440.83 | 0.00 | ORB-long ORB[439.00,443.35] vol=2.8x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 12:05:00 | 444.99 | 441.95 | 0.00 | T1 1.5R @ 444.99 |
| Target hit | 2025-05-23 15:20:00 | 446.40 | 444.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 09:35:00 | 450.90 | 451.90 | 0.00 | ORB-short ORB[451.10,453.50] vol=1.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-05-28 09:40:00 | 451.73 | 451.85 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 458.65 | 456.48 | 0.00 | ORB-long ORB[453.15,457.50] vol=6.0x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 09:40:00 | 460.78 | 457.71 | 0.00 | T1 1.5R @ 460.78 |
| Stop hit — per-position SL triggered | 2025-05-29 09:45:00 | 458.65 | 457.89 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 10:20:00 | 453.30 | 456.03 | 0.00 | ORB-short ORB[455.00,457.65] vol=1.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-06-02 10:40:00 | 455.06 | 455.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:25:00 | 471.90 | 470.10 | 0.00 | ORB-long ORB[467.30,470.00] vol=3.3x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-06-05 10:50:00 | 470.45 | 470.45 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:40:00 | 440.30 | 437.63 | 0.00 | ORB-long ORB[434.75,440.00] vol=1.5x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-06-20 09:50:00 | 438.39 | 437.95 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:25:00 | 440.90 | 442.75 | 0.00 | ORB-short ORB[442.05,445.00] vol=1.6x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:15:00 | 439.44 | 441.87 | 0.00 | T1 1.5R @ 439.44 |
| Stop hit — per-position SL triggered | 2025-06-26 12:50:00 | 440.90 | 441.28 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 11:00:00 | 446.65 | 448.99 | 0.00 | ORB-short ORB[449.20,452.50] vol=3.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-07-01 12:05:00 | 447.73 | 448.50 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:15:00 | 448.95 | 451.13 | 0.00 | ORB-short ORB[450.55,453.30] vol=1.8x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-07-02 11:20:00 | 450.00 | 450.67 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 09:40:00 | 441.20 | 442.61 | 0.00 | ORB-short ORB[442.10,445.55] vol=2.3x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:45:00 | 439.51 | 441.26 | 0.00 | T1 1.5R @ 439.51 |
| Target hit | 2025-07-07 15:20:00 | 437.25 | 439.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:10:00 | 435.45 | 437.36 | 0.00 | ORB-short ORB[437.00,439.45] vol=1.9x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-07-08 12:30:00 | 436.27 | 436.83 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 10:00:00 | 432.65 | 434.82 | 0.00 | ORB-short ORB[434.75,437.30] vol=2.0x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:55:00 | 431.28 | 433.51 | 0.00 | T1 1.5R @ 431.28 |
| Target hit | 2025-07-09 13:25:00 | 426.20 | 425.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2025-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:35:00 | 427.50 | 425.46 | 0.00 | ORB-long ORB[422.20,427.35] vol=2.0x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:40:00 | 429.04 | 426.96 | 0.00 | T1 1.5R @ 429.04 |
| Stop hit — per-position SL triggered | 2025-07-11 10:00:00 | 427.50 | 427.62 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 432.30 | 429.91 | 0.00 | ORB-long ORB[426.40,431.25] vol=1.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:55:00 | 434.13 | 430.84 | 0.00 | T1 1.5R @ 434.13 |
| Stop hit — per-position SL triggered | 2025-07-14 10:00:00 | 432.30 | 430.97 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:00:00 | 440.45 | 436.88 | 0.00 | ORB-long ORB[435.25,437.90] vol=1.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 439.36 | 438.13 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:15:00 | 434.30 | 435.34 | 0.00 | ORB-short ORB[435.00,438.30] vol=6.4x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-07-16 11:20:00 | 435.13 | 435.38 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:35:00 | 441.25 | 438.72 | 0.00 | ORB-long ORB[434.80,440.90] vol=2.5x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 11:00:00 | 443.25 | 440.05 | 0.00 | T1 1.5R @ 443.25 |
| Target hit | 2025-07-21 15:20:00 | 444.50 | 442.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2025-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:00:00 | 444.15 | 444.95 | 0.00 | ORB-short ORB[444.75,446.95] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-07-22 11:35:00 | 445.01 | 444.86 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:45:00 | 441.50 | 443.74 | 0.00 | ORB-short ORB[443.50,445.65] vol=1.9x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-07-25 09:50:00 | 442.50 | 443.58 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:40:00 | 417.85 | 420.08 | 0.00 | ORB-short ORB[418.80,422.90] vol=2.3x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 10:10:00 | 415.85 | 418.95 | 0.00 | T1 1.5R @ 415.85 |
| Stop hit — per-position SL triggered | 2025-08-11 10:30:00 | 417.85 | 418.51 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:55:00 | 430.95 | 428.18 | 0.00 | ORB-long ORB[426.80,429.95] vol=1.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-08-20 11:00:00 | 430.05 | 428.26 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:55:00 | 435.95 | 438.19 | 0.00 | ORB-short ORB[437.70,441.80] vol=10.2x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-09-05 11:10:00 | 437.16 | 437.41 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:55:00 | 439.20 | 437.20 | 0.00 | ORB-long ORB[433.55,437.85] vol=1.9x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-09-11 10:00:00 | 438.27 | 437.30 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:35:00 | 468.60 | 464.73 | 0.00 | ORB-long ORB[463.25,468.00] vol=1.7x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-09-15 10:45:00 | 466.97 | 465.06 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:40:00 | 453.00 | 455.79 | 0.00 | ORB-short ORB[455.30,460.10] vol=2.3x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-09-18 09:55:00 | 454.46 | 455.09 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 452.70 | 455.39 | 0.00 | ORB-short ORB[454.50,457.45] vol=2.1x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-09-19 11:25:00 | 453.47 | 455.05 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:40:00 | 462.20 | 459.52 | 0.00 | ORB-long ORB[455.55,460.50] vol=1.5x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-09-22 09:50:00 | 461.05 | 460.32 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:50:00 | 456.00 | 457.71 | 0.00 | ORB-short ORB[457.20,462.55] vol=2.1x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-09-23 11:05:00 | 457.23 | 457.24 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:55:00 | 457.05 | 458.69 | 0.00 | ORB-short ORB[457.10,463.90] vol=3.2x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 12:45:00 | 455.28 | 457.93 | 0.00 | T1 1.5R @ 455.28 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 457.05 | 457.62 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:55:00 | 460.60 | 457.74 | 0.00 | ORB-long ORB[453.30,459.25] vol=2.4x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-09-25 10:35:00 | 459.07 | 458.77 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:30:00 | 459.80 | 457.33 | 0.00 | ORB-long ORB[453.80,459.70] vol=1.6x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:00:00 | 462.50 | 459.13 | 0.00 | T1 1.5R @ 462.50 |
| Target hit | 2025-09-29 11:45:00 | 460.50 | 460.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2025-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:50:00 | 486.85 | 483.42 | 0.00 | ORB-long ORB[478.90,484.85] vol=2.2x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:05:00 | 489.77 | 485.05 | 0.00 | T1 1.5R @ 489.77 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 486.85 | 485.31 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:05:00 | 487.30 | 490.12 | 0.00 | ORB-short ORB[488.30,493.25] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-10-07 13:05:00 | 488.88 | 488.78 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 489.40 | 492.77 | 0.00 | ORB-short ORB[491.05,495.90] vol=2.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 490.93 | 492.30 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:15:00 | 487.20 | 482.41 | 0.00 | ORB-long ORB[477.30,483.85] vol=1.6x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-10-23 10:25:00 | 485.54 | 482.59 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 490.85 | 488.44 | 0.00 | ORB-long ORB[482.20,489.10] vol=2.2x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 09:50:00 | 493.50 | 490.96 | 0.00 | T1 1.5R @ 493.50 |
| Target hit | 2025-10-24 10:40:00 | 492.00 | 492.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2025-10-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 10:35:00 | 483.35 | 486.21 | 0.00 | ORB-short ORB[485.20,489.60] vol=3.3x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-10-27 12:20:00 | 484.78 | 485.15 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:00:00 | 475.30 | 476.68 | 0.00 | ORB-short ORB[476.00,480.15] vol=1.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-11-04 10:25:00 | 476.23 | 476.46 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:15:00 | 467.25 | 469.13 | 0.00 | ORB-short ORB[469.30,474.00] vol=2.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-11-06 11:55:00 | 468.29 | 468.92 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 461.00 | 463.62 | 0.00 | ORB-short ORB[461.70,466.85] vol=2.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-11-07 09:40:00 | 462.19 | 463.18 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 482.15 | 479.39 | 0.00 | ORB-long ORB[476.05,479.95] vol=3.2x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:45:00 | 484.42 | 481.83 | 0.00 | T1 1.5R @ 484.42 |
| Stop hit — per-position SL triggered | 2025-11-10 10:05:00 | 482.15 | 482.50 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:30:00 | 496.15 | 492.69 | 0.00 | ORB-long ORB[486.60,493.70] vol=5.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:40:00 | 498.86 | 494.68 | 0.00 | T1 1.5R @ 498.86 |
| Target hit | 2025-11-13 10:35:00 | 497.20 | 497.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:15:00 | 481.80 | 484.89 | 0.00 | ORB-short ORB[483.50,487.95] vol=1.9x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-11-17 11:30:00 | 482.83 | 484.70 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:05:00 | 469.45 | 474.31 | 0.00 | ORB-short ORB[474.20,480.95] vol=2.3x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-11-18 10:35:00 | 470.96 | 473.02 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 468.05 | 466.86 | 0.00 | ORB-long ORB[463.20,468.00] vol=1.8x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-11-26 13:25:00 | 466.79 | 467.95 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 474.55 | 476.55 | 0.00 | ORB-short ORB[475.15,478.50] vol=1.7x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-11-27 12:00:00 | 475.98 | 475.57 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 483.00 | 480.87 | 0.00 | ORB-long ORB[476.90,482.40] vol=2.5x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-11-28 09:35:00 | 481.62 | 480.95 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:35:00 | 494.75 | 497.61 | 0.00 | ORB-short ORB[495.75,501.70] vol=1.6x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-12-01 10:20:00 | 496.92 | 497.23 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 10:15:00 | 504.70 | 501.58 | 0.00 | ORB-long ORB[496.55,503.50] vol=2.0x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:35:00 | 507.01 | 503.39 | 0.00 | T1 1.5R @ 507.01 |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 504.70 | 504.15 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:55:00 | 544.75 | 538.85 | 0.00 | ORB-long ORB[532.35,539.75] vol=1.8x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 11:00:00 | 547.73 | 539.77 | 0.00 | T1 1.5R @ 547.73 |
| Stop hit — per-position SL triggered | 2025-12-12 11:10:00 | 544.75 | 540.14 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:20:00 | 591.40 | 583.96 | 0.00 | ORB-long ORB[576.85,582.50] vol=2.9x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-12-18 10:45:00 | 589.17 | 586.23 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:40:00 | 613.50 | 611.68 | 0.00 | ORB-long ORB[606.50,613.40] vol=2.6x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-12-23 11:25:00 | 611.59 | 612.52 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 625.20 | 621.68 | 0.00 | ORB-long ORB[617.95,623.20] vol=3.5x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 09:35:00 | 628.41 | 623.47 | 0.00 | T1 1.5R @ 628.41 |
| Target hit | 2025-12-24 11:40:00 | 625.85 | 626.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — BUY (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 622.35 | 619.38 | 0.00 | ORB-long ORB[614.30,620.40] vol=3.6x ATR=2.15 |
| Stop hit — per-position SL triggered | 2026-01-02 09:55:00 | 620.20 | 619.91 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-01-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:55:00 | 642.15 | 639.51 | 0.00 | ORB-long ORB[634.60,641.45] vol=2.1x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:25:00 | 645.28 | 641.17 | 0.00 | T1 1.5R @ 645.28 |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 642.15 | 642.95 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:35:00 | 639.65 | 646.58 | 0.00 | ORB-short ORB[644.85,650.75] vol=2.5x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-01-07 09:40:00 | 642.08 | 646.00 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 09:40:00 | 616.70 | 620.23 | 0.00 | ORB-short ORB[617.65,625.00] vol=1.7x ATR=3.01 |
| Stop hit — per-position SL triggered | 2026-01-12 09:45:00 | 619.71 | 620.12 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-01-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 09:30:00 | 646.95 | 641.77 | 0.00 | ORB-long ORB[636.45,644.70] vol=3.0x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-01-14 09:35:00 | 644.51 | 642.67 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 584.65 | 586.24 | 0.00 | ORB-short ORB[585.10,590.00] vol=1.6x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 586.44 | 586.05 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 615.25 | 609.70 | 0.00 | ORB-long ORB[602.85,609.80] vol=1.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-02-25 10:50:00 | 613.15 | 610.29 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 590.75 | 593.27 | 0.00 | ORB-short ORB[592.30,596.80] vol=1.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 592.60 | 592.81 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:15:00 | 514.55 | 519.12 | 0.00 | ORB-short ORB[517.20,524.95] vol=7.3x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-03-19 13:30:00 | 516.94 | 517.33 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:40:00 | 486.25 | 490.82 | 0.00 | ORB-short ORB[492.00,499.00] vol=1.5x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-03-24 11:20:00 | 488.41 | 489.76 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 618.25 | 620.40 | 0.00 | ORB-short ORB[618.65,625.55] vol=1.6x ATR=3.00 |
| Stop hit — per-position SL triggered | 2026-05-06 11:45:00 | 621.25 | 620.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:30:00 | 436.70 | 2025-05-14 09:35:00 | 438.46 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-05-14 09:30:00 | 436.70 | 2025-05-14 09:50:00 | 436.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 09:30:00 | 443.60 | 2025-05-15 09:35:00 | 442.02 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-21 11:00:00 | 440.80 | 2025-05-21 11:30:00 | 439.53 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-23 11:00:00 | 443.50 | 2025-05-23 12:05:00 | 444.99 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-23 11:00:00 | 443.50 | 2025-05-23 15:20:00 | 446.40 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2025-05-28 09:35:00 | 450.90 | 2025-05-28 09:40:00 | 451.73 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-05-29 09:30:00 | 458.65 | 2025-05-29 09:40:00 | 460.78 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-05-29 09:30:00 | 458.65 | 2025-05-29 09:45:00 | 458.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-02 10:20:00 | 453.30 | 2025-06-02 10:40:00 | 455.06 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-05 10:25:00 | 471.90 | 2025-06-05 10:50:00 | 470.45 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-20 09:40:00 | 440.30 | 2025-06-20 09:50:00 | 438.39 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-06-26 10:25:00 | 440.90 | 2025-06-26 11:15:00 | 439.44 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-06-26 10:25:00 | 440.90 | 2025-06-26 12:50:00 | 440.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 11:00:00 | 446.65 | 2025-07-01 12:05:00 | 447.73 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-02 10:15:00 | 448.95 | 2025-07-02 11:20:00 | 450.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-07 09:40:00 | 441.20 | 2025-07-07 11:45:00 | 439.51 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-07 09:40:00 | 441.20 | 2025-07-07 15:20:00 | 437.25 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2025-07-08 11:10:00 | 435.45 | 2025-07-08 12:30:00 | 436.27 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-09 10:00:00 | 432.65 | 2025-07-09 10:55:00 | 431.28 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-09 10:00:00 | 432.65 | 2025-07-09 13:25:00 | 426.20 | TARGET_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2025-07-11 09:35:00 | 427.50 | 2025-07-11 09:40:00 | 429.04 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-07-11 09:35:00 | 427.50 | 2025-07-11 10:00:00 | 427.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 09:40:00 | 432.30 | 2025-07-14 09:55:00 | 434.13 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-14 09:40:00 | 432.30 | 2025-07-14 10:00:00 | 432.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:00:00 | 440.45 | 2025-07-15 10:15:00 | 439.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-16 11:15:00 | 434.30 | 2025-07-16 11:20:00 | 435.13 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-21 10:35:00 | 441.25 | 2025-07-21 11:00:00 | 443.25 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-07-21 10:35:00 | 441.25 | 2025-07-21 15:20:00 | 444.50 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2025-07-22 11:00:00 | 444.15 | 2025-07-22 11:35:00 | 445.01 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-25 09:45:00 | 441.50 | 2025-07-25 09:50:00 | 442.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-11 09:40:00 | 417.85 | 2025-08-11 10:10:00 | 415.85 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-08-11 09:40:00 | 417.85 | 2025-08-11 10:30:00 | 417.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 10:55:00 | 430.95 | 2025-08-20 11:00:00 | 430.05 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-05 10:55:00 | 435.95 | 2025-09-05 11:10:00 | 437.16 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-11 09:55:00 | 439.20 | 2025-09-11 10:00:00 | 438.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-15 10:35:00 | 468.60 | 2025-09-15 10:45:00 | 466.97 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-18 09:40:00 | 453.00 | 2025-09-18 09:55:00 | 454.46 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-19 11:15:00 | 452.70 | 2025-09-19 11:25:00 | 453.47 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-22 09:40:00 | 462.20 | 2025-09-22 09:50:00 | 461.05 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-23 10:50:00 | 456.00 | 2025-09-23 11:05:00 | 457.23 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-24 10:55:00 | 457.05 | 2025-09-24 12:45:00 | 455.28 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-09-24 10:55:00 | 457.05 | 2025-09-24 13:15:00 | 457.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-25 09:55:00 | 460.60 | 2025-09-25 10:35:00 | 459.07 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-29 09:30:00 | 459.80 | 2025-09-29 10:00:00 | 462.50 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-09-29 09:30:00 | 459.80 | 2025-09-29 11:45:00 | 460.50 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-10-03 09:50:00 | 486.85 | 2025-10-03 10:05:00 | 489.77 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-10-03 09:50:00 | 486.85 | 2025-10-03 10:15:00 | 486.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-07 11:05:00 | 487.30 | 2025-10-07 13:05:00 | 488.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-08 11:00:00 | 489.40 | 2025-10-08 11:25:00 | 490.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-23 10:15:00 | 487.20 | 2025-10-23 10:25:00 | 485.54 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-24 09:30:00 | 490.85 | 2025-10-24 09:50:00 | 493.50 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-10-24 09:30:00 | 490.85 | 2025-10-24 10:40:00 | 492.00 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-10-27 10:35:00 | 483.35 | 2025-10-27 12:20:00 | 484.78 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-04 10:00:00 | 475.30 | 2025-11-04 10:25:00 | 476.23 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-06 11:15:00 | 467.25 | 2025-11-06 11:55:00 | 468.29 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-07 09:30:00 | 461.00 | 2025-11-07 09:40:00 | 462.19 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-10 09:30:00 | 482.15 | 2025-11-10 09:45:00 | 484.42 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-11-10 09:30:00 | 482.15 | 2025-11-10 10:05:00 | 482.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 09:30:00 | 496.15 | 2025-11-13 09:40:00 | 498.86 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-11-13 09:30:00 | 496.15 | 2025-11-13 10:35:00 | 497.20 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-11-17 11:15:00 | 481.80 | 2025-11-17 11:30:00 | 482.83 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-18 10:05:00 | 469.45 | 2025-11-18 10:35:00 | 470.96 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-26 09:30:00 | 468.05 | 2025-11-26 13:25:00 | 466.79 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-27 09:30:00 | 474.55 | 2025-11-27 12:00:00 | 475.98 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-28 09:30:00 | 483.00 | 2025-11-28 09:35:00 | 481.62 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-01 09:35:00 | 494.75 | 2025-12-01 10:20:00 | 496.92 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-12-03 10:15:00 | 504.70 | 2025-12-03 10:35:00 | 507.01 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-03 10:15:00 | 504.70 | 2025-12-03 11:15:00 | 504.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 10:55:00 | 544.75 | 2025-12-12 11:00:00 | 547.73 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-12-12 10:55:00 | 544.75 | 2025-12-12 11:10:00 | 544.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-18 10:20:00 | 591.40 | 2025-12-18 10:45:00 | 589.17 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-12-23 09:40:00 | 613.50 | 2025-12-23 11:25:00 | 611.59 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-24 09:30:00 | 625.20 | 2025-12-24 09:35:00 | 628.41 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-12-24 09:30:00 | 625.20 | 2025-12-24 11:40:00 | 625.85 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-01-02 09:35:00 | 622.35 | 2026-01-02 09:55:00 | 620.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-01-06 09:55:00 | 642.15 | 2026-01-06 10:25:00 | 645.28 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-01-06 09:55:00 | 642.15 | 2026-01-06 12:15:00 | 642.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 09:35:00 | 639.65 | 2026-01-07 09:40:00 | 642.08 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-01-12 09:40:00 | 616.70 | 2026-01-12 09:45:00 | 619.71 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-01-14 09:30:00 | 646.95 | 2026-01-14 09:35:00 | 644.51 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-17 09:35:00 | 584.65 | 2026-02-17 09:50:00 | 586.44 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-25 10:25:00 | 615.25 | 2026-02-25 10:50:00 | 613.15 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-06 10:45:00 | 590.75 | 2026-03-06 11:35:00 | 592.60 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-19 11:15:00 | 514.55 | 2026-03-19 13:30:00 | 516.94 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-24 10:40:00 | 486.25 | 2026-03-24 11:20:00 | 488.41 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-05-06 11:05:00 | 618.25 | 2026-05-06 11:45:00 | 621.25 | STOP_HIT | 1.00 | -0.49% |
